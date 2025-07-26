import os
import json
import torch
import argparse
import math
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Literal
from transformers import AutoProcessor, GenerationConfig
from model_utils.wrapped_visualizer import AnoleforConditionalGeneration

def main(args: argparse.Namespace):
    """生成和处理模型输出的主函数"""
    # 使用提供的load_model函数加载模型和处理器
    model_data = load_model(args)
    model = model_data['model']
    processor = model_data['processor']
    
    print(f"模型已加载: {args.model}")
    if args.model_ckpt:
        print(f"使用检查点: {args.model_ckpt}")
    

    
    # 处理输入文本和图像
    input_text = args.input
    images = []
    image_paths = ['/home/baiqiao/spatial_generation/Spatial_Mental_Mani/data/Goal_Imgs/000_1.png','/home/baiqiao/spatial_generation/Spatial_Mental_Mani/data/candidate_blocks_images/000_16_cand.png']
    for img_path in image_paths:
        input_text += "<image>"  # 图像占位符
        img = Image.open(img_path).convert('RGB')
        images.append(img)
    
    # 处理图像(如果有)
    if images:
        pixel_values = processor(images=images, text=input_text, return_tensors="pt").pixel_values
    else:
        pixel_values = None
    
    # 对输入文本进行分词
    tokenized_input = processor(
        text=input_text,
        padding="max_length",
        return_tensors="pt",
        max_length=args.max_length
    )
    
    # 将输入移至模型设备
    tokenized_input = {k: v.to(model.device) for k, v in tokenized_input.items()}
    if pixel_values is not None:
        pixel_values = pixel_values.to(model.device).to(torch.bfloat16)
    
    # 创建输出目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置生成配置
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    # 使用recursive_generate生成输出
    generated_text, updated_text, all_images, all_pil_images = model.recursive_generate(
        processor=processor,
        input_text=input_text,
        save_dir=args.save_dir,
        generation_config=generation_config,
        multimodal_generation_mode="interleaved-text-image",
        input_ids=tokenized_input['input_ids'],
        attention_mask=tokenized_input['attention_mask'],
        pixel_values=pixel_values,
    )
    
    # 打印结果
    # 保存所有生成的图像
    for i, pil_img in enumerate(all_pil_images):
        image_path = os.path.join(args.save_dir, f"{i}.jpg")
        pil_img.save(image_path)
        print(f"图像 {i+1} 已保存到 {image_path}")
    print("\n生成内容:")
    print("=" * 50)
    print(updated_text)
    print("=" * 50)
    print(f"生成了 {len(all_pil_images)} 张图像，保存至 {args.save_dir}")
    
    # 保存完整输出为JSON文件
    output_data = {
        "input_text": input_text,
        "generated_text": updated_text,
        "image_paths": [os.path.join(args.save_dir, f"{i}.jpg") for i in range(len(all_pil_images))]
    }
    
    with open(os.path.join(args.save_dir, "output.json"), "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"完整输出已保存到 {os.path.join(args.save_dir, 'output.json')}")

def load_model(args):
    model_name = args.model

    model_ckpt_path = args.model_ckpt

    if model_name in ['anole']:
        image_token_num = args.image_seq_length

        model = AnoleforConditionalGeneration.from_pretrained(
            "leloy/Anole-7b-v0.1-hf",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            codebook_sim="mse"
        )
        processor = AutoProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf", image_seq_length=image_token_num)
        processor.image_processor.size = {"shortest_edge": int(512 / int(math.sqrt(1024 / image_token_num)))}
        processor.image_processor.crop_size = {
            "height": int(512 / int(math.sqrt(1024 / image_token_num))),
            "width": int(512 / int(math.sqrt(1024 / image_token_num)))
        }

        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.model.vqmodel.config.resolution = processor.image_processor.size["shortest_edge"]
        model.model.vqmodel.quantize.quant_state_dims = [
            model.model.vqmodel.config.resolution // 2 ** (len(model.model.vqmodel.config.channel_multiplier) - 1)
        ] * 2

        args.sketch_resolution = model.model.vqmodel.config.resolution
        model.sketch_resolution = (args.sketch_resolution, args.sketch_resolution)
        model.image_token_num = image_token_num

        model.get_vis_codebook_sim()

        from peft import LoraConfig, get_peft_model
        from peft.peft_model import PeftModel

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=['q_proj', "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["lm_head"],
            inference_mode=True
        )
        lora_model = get_peft_model(model, config)

        if args.do_eval and not args.do_train and model_ckpt_path:
            lora_model.load_adapter(model_ckpt_path, 'default', is_trainable=False)

        return {
            'processor': processor,
            'model': lora_model
        }
    else:
        raise ValueError("Unsupported model type. ")

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用Anole模型生成交错的图像-文本内容")
    parser.add_argument("-i", "--input", type=str, required=True, help="多模态输入文件")
    parser.add_argument("-s", "--save_dir", type=str, default="./output/inference/", help="保存生成图像的目录")
    parser.add_argument("-m", "--model", type=str, default="anole", help="模型类型")
    parser.add_argument("--model_ckpt", type=str, default=None, help="模型检查点路径")
    parser.add_argument("--image_seq_length", type=int, default=300, help="图像序列长度")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="生成的最大新标记数")
    parser.add_argument("--max_length", type=int, default=1500, help="输入序列的最大长度")
    parser.add_argument("--temperature", type=float, default=0.1, help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.7, help="top-p采样参数")
    parser.add_argument("--do_eval", action="store_true", help="执行评估")
    parser.add_argument("--do_train", action="store_false", help="执行训练")
    
    args: argparse.Namespace = parser.parse_args()
    return args

if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    main(args)
# python utils/inference_ybq_anole.py -i 'Please generate a two-step visual guide for building the shape in the first image, using the blocks shown in the second image.'