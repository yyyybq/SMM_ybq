import os
import wandb
import torch
import logging
import argparse
import yaml
import copy
import configparser
import transformers
import torch.distributed as dist

from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback, StopStringCriteria, set_seed, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from transformers.data.data_collator import default_data_collator
from transformers.trainer_utils import get_last_checkpoint
from transformers.generation import StoppingCriteriaList

from utils.run_config import create_run_name
from utils.training_arguments import WrappedSeq2SeqTrainingArguments
from utils.load_data import load_data, tokenize_dataset
from utils.load_model import load_model
from utils.evaluator import VisualizationEvaluator

logger = logging.getLogger(__name__)

WANDB_API_KEY = "<YOUR_WANDB_KEY_API>"
WANDB_ENTITY = "<YOUR_WANDB_ENTITY>"
PROJECT_NAME = "<YOUR_PROJECT_NAME>"

def init(args):
    os.environ[
        'CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if args.local_rank is not None:
        torch.cuda.set_device(args.local_rank)
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)

    # Read in the training arguments
    setting_type = "interleaved"
    with open(os.path.join(args.cfg_path, setting_type + '.yaml')) as f:
        file = f.read()
        training_cfg = yaml.safe_load(file)

    if args.train_bz:
        training_cfg['hyper']['train_batch_size'] = args.train_bz
    if args.val_bz:
        training_cfg['hyper']['val_batch_size'] = args.val_bz
    if args.grad_acc:
        training_cfg['hyper']['grad_accumulation'] = args.grad_acc

    sup_hyper = training_cfg["hyper"]

    # Construct the run_name of the task
    args.run_name = create_run_name(args, training_cfg)

    args.run_name = args.note + args.run_name

    training_args = WrappedSeq2SeqTrainingArguments(
        output_dir=os.path.join(args.output, args.run_name),
        remove_unused_columns=False,
        evaluation_strategy=training_cfg['eval']['eval_strategy'],
        eval_steps=training_cfg['eval']['eval_steps'] if training_cfg['eval']['eval_strategy'] == "steps" else None,
        save_strategy=training_cfg['save']['save_strategy'],
        save_steps=training_cfg['save']['save_steps'] if training_cfg['save']['save_strategy'] == "steps" else None,
        save_total_limit=40,
        seed=args.seed,
        # note: for supervised tuning
        #############################
        learning_rate=sup_hyper['lr'] if sup_hyper else 0,
        per_device_train_batch_size=sup_hyper['train_batch_size'] if sup_hyper else 0,
        gradient_accumulation_steps=sup_hyper['grad_accumulation'] if sup_hyper else 0,
        per_device_eval_batch_size=sup_hyper['val_batch_size'] if sup_hyper else training_cfg['hyper']['val_batch_size'],
        num_train_epochs=sup_hyper['epochs'] if sup_hyper else 0,
        #############################
        # warmup_ratio=0.1,
        logging_steps=training_cfg['logging']['logging_step'],
        push_to_hub=False,
        # customize
        predict_with_generate=training_cfg['model']['predict_with_generate'],
        generation_max_new_tokens=training_cfg['model']['generation_max_new_tokens'],
        generation_num_beams=training_cfg['model']['generation_num_beams']
    )

    # Initialize the wandb logger if specified
    try:
        rank = dist.get_rank()
    except:
        rank = 0
    args.local_rank = rank

    if args.report_to == "wandb" and rank == 0:
        import wandb
        init_args = {}

        # note: my new wandb api key
        wandb.login(key=WANDB_API_KEY)

        if "MLFLOW_EXPERIMENT_ID" in os.environ:
            init_args["group"] = os.environ["MLFLOW_EXPERIMENT_ID"]
        
        if args.local_rank == 0 or args.local_rank is None:
            wandb.init(
                project=os.getenv("WANDB_PROJECT", PROJECT_NAME),
                name=args.run_name,
                entity=os.getenv("WANDB_ENTITY", WANDB_ENTITY),
                **init_args,
            )
            wandb.config.update(training_args, allow_val_change=True)
    else:
        training_args.report_to = []

    if os.path.exists(training_args.output_dir):
        args.model_ckpt = training_args.output_dir

    # Detect the checkpoint
    if args.model_ckpt is not None:
        training_args.load_weights_from = get_last_checkpoint(args.model_ckpt)
    else:
        training_args.load_weights_from = None

    return training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="anole")
    parser.add_argument("--data", type=str, nargs="+")
    parser.add_argument("--data_dir", type=str, default="data_samples")
    parser.add_argument("--decoder_type", type=str, default='anole')
    parser.add_argument('--note', type=str, default="debug")
    parser.add_argument('--image_seq_length', type=int, default=1024)
    parser.add_argument('--no_perceptual_loss', action="store_true")

    # model argument
    parser.add_argument('--model_ckpt', type=str, default=None, help='path of the checkpoint')
    parser.add_argument('--load_last_checkpoint', action='store_true')

    # training arguments
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--cfg_path', type=str, default='cfg')
    parser.add_argument('--patience', type=int, default=5)

    # input format argument
    parser.add_argument('--input_format', type=str, default="anole")

    # output configuration
    parser.add_argument('--output', type=str, default='outputs')
    parser.add_argument('--report_to', type=str, default="wandb")
    parser.add_argument('--cache_dir', type=str, default=None)

    # randomness
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local_rank", type=int)

    # debug
    parser.add_argument('--toy', action='store_true')

    # shortcut customization
    parser.add_argument('--train_bz', type=int, default=None)
    parser.add_argument('--val_bz', type=int, default=None)
    parser.add_argument('--grad_acc', type=int, default=None)

    args = parser.parse_args()

    if args.model in ['anole']:
        args.decoder_type = args.model
        assert args.input_format == "anole"

    if args.decoder_type in ['anole']:
        args.note = args.note + f"image_seq_len-{str(args.image_seq_length)}-"

    training_args = init(args)

    print(f'Preparing the {args.data} dataset... ')
    data = load_data(dataset=args.data, data_dir=args.data_dir)

    if len(data) == 2:
        train_split, eval_split, test_split = data['train'], None, data['test']
    else:
        try:
            train_split, eval_split, test_split = data['train'], data['dev'], data['test']
        except:
            train_split, eval_split, test_split = data['train'], data['validation'], data['test']

    if args.toy:
        print('Only using toy examples for debugging...')
        train_split = train_split.select(list(range(100)))
        if eval_split:
            eval_split = eval_split.select(list(range(10)))
        test_split = test_split.select(list(range(10)))

    model_processor = load_model(args)
    model, processor = model_processor['model'], model_processor["processor"]
    
    eval_data_num = (len(eval_split) // (training_args.per_device_eval_batch_size * torch.cuda.device_count())) * (training_args.per_device_eval_batch_size * torch.cuda.device_count())
    eval_split = eval_split.select(list(range(eval_data_num)))
    test_data_num = (len(test_split) // (training_args.per_device_eval_batch_size * torch.cuda.device_count())) * (training_args.per_device_eval_batch_size * torch.cuda.device_count())
    test_split = test_split.select(list(range(test_data_num)))

    print(f"Eval Num: {eval_data_num}")

    tokenized_data, max_source_length, max_target_length = tokenize_dataset(
        train_split=train_split,
        eval_split=eval_split,
        test_split=test_split,
        model=model,
        processor=processor,
        input_format=args.input_format,
        interleave=True,
        data_name = "-".join(args.data),
    )

    training_args.generation_max_new_tokens = max_target_length + 100
    print(f"generation_max_new_tokens: {training_args.generation_max_new_tokens}")

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.patience)
    label_pad_token_id = -100
    
    # Data collator: 
    from utils.data_collator import customize_data_collator
    data_collator = customize_data_collator
    
    from utils.trainer.customize_trainer import CustomizeSeq2SeqTrainer
    trainer_type = CustomizeSeq2SeqTrainer

    # fixme:
    training_args.label_smoothing_factor = 0.1

    if args.model in ['anole']:
        # used in evaluation when do_eval
        kwargs = dict()
        kwargs['multimodal_generation_mode'] = "interleaved-text-image"     # see L217 in wrapped_visualizer.py
        kwargs['stopping_criteria'] = StoppingCriteriaList([StopStringCriteria(stop_strings=["<reserved08706>", "</s>"], tokenizer=processor.tokenizer)])
        # used in evaluation during training
        training_args.customize_gen_stopping_criteria = StoppingCriteriaList([StopStringCriteria(stop_strings=["<reserved08706>", "</s>"], tokenizer=processor.tokenizer)])

    trainer = trainer_type(
        args=training_args,
        model=model,
        evaluator=VisualizationEvaluator(args=args),
        # We name it "evaluator" while the hugging face call it "Metric",
        # they are all f(predictions: List, references: List of dict) = eval_result: dict
        tokenizer=processor,
        data_collator=data_collator,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['eval'] if 'eval' in tokenized_data.keys() else tokenized_data['test'],
        eval_examples=eval_split if 'eval' in tokenized_data.keys() else test_split,
        wandb_run_dir=wandb.run.dir if "wandb" in training_args.report_to and training_args.local_rank <= 0 else None,
        # callbacks=[early_stopping_callback],  # currently disabled early stopping for now
        image_loss_func=not args.no_perceptual_loss, 
    )

    print('Trainer build successfully.')

    # for anole, there would be different inference mode. We use kwargs to pass these settings into the inference process.
    checkpoint = None
    if training_args.load_weights_from is not None:
        checkpoint = training_args.load_weights_from

    # NOTE: train the model with supervision
    if args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = len(tokenized_data['train'])
        metrics["train_samples"] = min(max_train_samples, len(tokenized_data['train']))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            **kwargs
        )
        max_eval_samples = len(tokenized_data['eval'])
        metrics["eval_samples"] = min(max_eval_samples, len(tokenized_data['eval']))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset=tokenized_data['test'],
            test_examples=tokenized_data['test'].dataset,
            metric_key_prefix="predict",
            **kwargs
        )
        metrics = predict_results.metrics
        max_predict_samples = len(tokenized_data['test'])
        metrics["predict_samples"] = min(max_predict_samples, len(tokenized_data['test']))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)