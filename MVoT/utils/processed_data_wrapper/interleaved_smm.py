# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import ast
import shutil

import datasets
import string

from PIL import Image, ImageOps

_DATA_DIR = r"maze"

_URLS = {
    "data_dir": _DATA_DIR,
}

SINGLE_STEP_VISUALIZATION_INSTRUCTION = {
    "smm_simulation": "<INIT_STATE>\nResponse: <ACTION_HISTORY>"
}
LONG_HORIZON_VISUALIZATION_INSTRUCTION = {
    "smm_simulation": "<INIT_STATE>\nResponse: <ACTION_HISTORY>"
}
REAL_GOAL_INSTRUCTION = {
    "smm_simulation": "Task: Building Blocks Simulation\nDetermine the final destination (A, B, C or D) from the starting point (red point) following the action sequence. The definitions of the actions are as below. \n* Go up/left/down/right: move one grid space in the absolute up/left/down/right direction. \nFull Action Sequence: <ACTION_SEQ>\n"
}

import re


 


class MazeConfig(datasets.BuilderConfig):
    """BuilderConfig for Maze."""

    def __init__(self, tasks, modes, data_dir, **kwargs):
        """BuilderConfig for Maze.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MazeConfig, self).__init__(**kwargs)
        self.tasks = tasks
        self.modes = modes
        self.data_dir = data_dir


class Maze(datasets.GeneratorBasedBuilder):
    """Maze dataset."""

    BUILDER_CONFIG_CLASS = MazeConfig
    BUILDER_CONFIGS = [
        MazeConfig(
            name="processed_smm",
            tasks=["simulation"],
            modes=["single_step_visualization", "action_reasoning"],
            data_dir="/lustre/fsw/portfolios/nvr/users/ymingli/projects/SMM_data/data/reorganized_by_view_per_folder"
        )
    ]

    DEFAULT_CONFIG_NAME = "processed_smm"

    def _info(self):
        features = datasets.Features(
            {
                'idx': datasets.Value('int32'),
                "input_text": datasets.Value("string"),
                "input_imgs": datasets.Sequence(datasets.Image()),
                "label_text": datasets.Value("string"),
                "label_imgs": datasets.Sequence(datasets.Image()),
                "label_img_paths": datasets.Sequence(datasets.Value("string")),
                "input_img_paths": datasets.Sequence(datasets.Value("string")),
                'task': datasets.Value('string'), 
                'train_task': datasets.Value("string")
            }
        )

        return datasets.DatasetInfo(
            features=features,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = _URLS

        tasks = self.config.tasks
        modes = self.config.modes
        data_dir = self.config.data_dir

        global _DATA_DIR_PREFIX
        _DATA_DIR_PREFIX = data_dir


        data_dirs=data_dir
                  
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dirs": data_dirs,
                    "split": "train",
                    "modes": modes
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split('dev'),
                gen_kwargs={
                    "data_dirs": data_dirs,
                    "split": "dev",
                    "modes": modes
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split('test'),
                gen_kwargs={
                    "data_dirs": data_dirs,
                    "split": "test",
                    "modes": modes
                },
            ),
        ]

    def _generate_examples(self, data_dirs: list, split: str, modes: list):
        all_entries = os.listdir(data_dirs)
        all_data = []
        # 获取母文件夹的名称，用于构建相对路径
        root_folder_name = os.path.basename(data_dirs)

        for entry in all_entries:
            if  'view_0' in entry and '232_' not in entry:
                id = entry.split("_")[0]


                entry_path = os.path.join(data_dirs, entry)
                image_paths = []
                # 判断当前 entry 是否为子文件夹
                if os.path.isdir(entry_path):
                    # 第2步：从子文件夹出发，遍历里面的所有文件
                    filenames = os.listdir(entry_path)
                    def sort_key(filename):
                        match = re.search(r'step(\d+)_', filename)
                        if match:
                            return int(match.group(1))
                        return -1 # 如果没有匹配到，放前面
                        
                    sorted_filenames = sorted(filenames, key=sort_key)
                    for filename in sorted_filenames:
                    # 构建文件的完整路径
                        file_path = os.path.join(entry_path, filename)
                        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                            # 构建图片的相对路径
                            # os.path.relpath(path, start) 可以生成从 start 到 path 的相对路径
                            relative_path = os.path.relpath(file_path, data_dirs)
                            image_paths.append(relative_path)
                text_file = os.path.join('./step_descrip',f"{id}_description.json")
                with open(text_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    descriptions = data.get("descriptions")
                if image_paths==[]:
                    print(entry)
                all_data += [{"image_paths":image_paths, "action_list":descriptions,"task": "smm_simulation"}]
            
        # all_data = []
        # for data_dir in data_dirs:
        #     all_datasets = os.listdir(data_dir)

        #     json_files = [i for i in all_datasets if i.endswith(".json")]

        #     for json_f in json_files:
        #         with open(os.path.join(data_dir, json_f)) as f:
        #             data = json.load(f)
        #             diff_data = remove_duplicates(data)

        #             train_num = int(len(diff_data) * 0.8)
                
        #         # FIXME: the training environments are set to the same as dev/test environment because this is just a toy dataset for debugging
        #         # To use the formal implementation to avoid data contamination, uncomment the following code blocks
                    
        #         # if split in ['train']:
        #         #     all_data += [{**item, "task": data_dir} for item in diff_data[:train_num]]
        #         # else:
        #         #     all_data += [{**item, "task": data_dir} for item in diff_data[train_num:]]

        #         all_data += [{**item, "task": "smm_simulation"} for item in diff_data]
        
        data_idx = 0
        for data_item in all_data:

            interleaved_data_list = get_interleaved_data_smm(
                data_item,
                mode=modes
            )
            
            for item in interleaved_data_list:
                return_info = {
                    'idx': data_idx,
                    "input_text": item['input_text'].replace("<ACTION_SEQ>", "".join([f"Go {i.split('-')[0]}. " for i in data_item['action_list']])),
                    "input_imgs": item["input_imgs"],
                    "label_text": item['label_text'],
                    "label_imgs": item['label_imgs'],
                    "label_img_paths": item['label_img_paths'],
                    "input_img_paths": item['input_img_paths'],
                    "task": 'smm',
                    "train_task": item['train_task']
                }
                yield data_idx, return_info
                data_idx += 1

def remove_duplicates(data_dict_list):
    past_paths = []
    unique_list = []
    for data_dict in data_dict_list:
        sublist = data_dict['path_locs']
        if sublist not in past_paths:
            past_paths.append(sublist)
            unique_list.append(data_dict)
    return unique_list


def get_interleaved_data_smm(data_item, mode=["single_step_visualization", "action_reasoning"]):
    """
    Args:
        image_paths: List of image paths showing states from start to end
        action_texts: List of action texts describing transitions between states
        mode: List of processing modes
    """
    interleaved_data = []
    image_paths = [os.path.join(_DATA_DIR_PREFIX, i) for i in data_item['image_paths']]
    action_texts = data_item['action_list']
    try:
        all_pil_images = [Image.open(img_path).convert("RGB").resize((256, 256)) for img_path in image_paths]
    except:
        return interleaved_data

    # Add a placeholder action for the initial state
    all_actions = ["initial_state"] + action_texts
    
    if "single_step_visualization" in mode:
        # Create batches for visualization

        image_batches = [image_paths[:2]] + [image_paths[:1] + image_paths[i:i+2] for i in range(1, len(image_paths)-1)]
        pil_image_batches = [all_pil_images[:2]] + [all_pil_images[:1] + all_pil_images[i:i+2] for i in range(1, len(all_pil_images)-1)]
        action_batches = [all_actions[:2]] + [all_actions[0:i+2] for i in range(1, len(all_actions)-1)]
        # print(image_paths)

        # print(image_batches)
        # print(pil_image_batches)
        for batch_idx, (image_batch, pil_image_batch, action_batch) in enumerate(zip(image_batches, pil_image_batches, action_batches)):
            input_image_paths = [image_batch[0], image_batch[-2]] if len(image_batch) != 2 else [image_batch[0]]
            label_image_paths = image_batch[-1:]

            # Construct text descriptions
            history_text = "".join(action_batch[1:-1]) + "<image>" + action_batch[-1] if len(action_batch) != 2 else action_batch[-1]
            init_state_text = action_batch[0] + "<image>"
            input_text = f"Given the initial state and action history, predict the next state. Initial state: {init_state_text}. Action history: {history_text}"

            input_imgs = pil_image_batch[:1] + pil_image_batch[-2:-1] if len(pil_image_batch) != 2 else pil_image_batch[:1]
            label_imgs = pil_image_batch[-1:]

            return_info = {
                "input_text": input_text,
                "label_text": "<image>",
                "input_imgs": input_imgs,
                "input_img_paths": input_image_paths,
                "label_imgs": label_imgs,
                "label_img_paths": label_image_paths,
                "task":data_item["task"],
                'train_task': "single_step_visualization"
            }
            interleaved_data.append(return_info)
    
    if "action_reasoning" in mode:
        # Create batches for action reasoning
        image_batches = [image_paths[:1]] + [image_paths[:1] + image_paths[i:i+1] for i in range(1, len(image_paths))]
        pil_image_batches = [all_pil_images[:1]] + [all_pil_images[:1] + all_pil_images[i:i+1] for i in range(1, len(all_pil_images))]
        action_batches = [all_actions[0:2]] + [all_actions[0:i+2] for i in range(1, len(all_actions))]

        for batch_idx, (image_batch, pil_image_batch, action_batch) in enumerate(zip(image_batches, pil_image_batches, action_batches)):
            input_image_paths = [image_batch[0], image_batch[-1]] if len(image_batch) != 1 else [image_batch[0]]
            label_image_paths = []

            # Construct text descriptions
            input_texts = action_batch[:-1] if batch_idx != (len(image_batches) - 1) else action_batch
            output_texts = action_batch[-1:] if batch_idx != (len(image_batches) - 1) else ["Action sequence completed."]

            history_text = "".join(input_texts[1:]) + "<image>" if batch_idx != 0 else ""
            init_state_text = input_texts[0] + "<image>"
            input_text = f"Predict the next action. Initial state: {init_state_text}. Action history: {history_text}"

            input_imgs = pil_image_batch[:1] + pil_image_batch[-1:] if batch_idx != 0 else pil_image_batch[:1]
            label_imgs = []

            return_info = {
                "input_text": input_text,
                "label_text": " ".join([o_t.strip() for o_t in output_texts]),
                "input_imgs": input_imgs,
                "input_img_paths": input_image_paths,
                "label_imgs": label_imgs,
                "label_img_paths": label_image_paths,
                "train_task": "action_reasoning"
            }
            interleaved_data.append(return_info)

    return interleaved_data

def get_text_from_actions(action_list):
    text_list = []
    for action in action_list:
        if action == "input_img":
            text_list.append("Initial smm: ")
        elif action == "start": 
            text_list.append("Start Point. ")
        else:
            meta_action = action.split("-")
            text_list.append(f"Go {meta_action[0]}. ")
    return text_list

def get_answer(data_item):
    if data_item['task'].endswith("simulation"):
        task = "simulation"

        options = [tuple(i) for i in data_item['candidate_locs']]
        destination_loc = tuple(data_item['path_locs'][-1])
        output_text = string.ascii_uppercase[options.index(destination_loc)]
    
    else:
        raise ValueError("Unsupported task. ")
    return output_text