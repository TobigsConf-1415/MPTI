import os
import logging
import glob
import pandas as pd
import argparse
import torch

from transformers import WEIGHTS_NAME, AutoConfig, AutoModelForCausalLM, AutoTokenizer

from train import train, evaluate
from utils import load_and_cache_examples, set_seed, _sorted_checkpoints

def main(args, logger, df_trn, df_val):
    
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
        and not args.should_continue
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    device = torch.device("cpu" if args.no_cuda else "cuda")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=args.cache_dir,
    )
    model.to(args.device)
    tokenizer.add_tokens(["<NAME>"], special_tokens=True)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, logger, tokenizer, df_trn, df_val, evaluate=False)

        global_step, tr_loss = train(args, logger, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForCausalLM.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = AutoModelForCausalLM.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, logger, model, tokenizer, df_trn, df_val, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', '-o', type=str, default = 'output')
    parser.add_argument('--model_type', '-mt', type=str, default = 'gpt2')
    parser.add_argument('--model_name_or_path', '-mn', type=str, default = 'microsoft/DialoGPT-medium')
    parser.add_argument('--config_name', '-cn', type=str, default = 'microsoft/DialoGPT-medium')
    parser.add_argument('--tokenizer_name', '-tn', type=str, default = 'microsoft/DialoGPT-medium')
    parser.add_argument('--cache_dir', '-cd', type=str, default = 'cached')
    parser.add_argument('--block_size', '-bs', type=int, default = 512)
    parser.add_argument('--do_train', type=bool, default = True)
    parser.add_argument('--do_eval', type=bool, default = True)
    parser.add_argument('--evaluate_during_training', type=bool, default = True)
    parser.add_argument('--per_gpu_train_batch_size', '-tbs', type=int, default = 1)
    parser.add_argument('--per_gpu_eval_batch_size', '-ebs', type=int, default = 1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default = 1)
    parser.add_argument('--learning_rate','-lr', type=float, default = 1e-7)
    parser.add_argument('--weight_decay', type=float, default = 0.0)
    parser.add_argument('--adam_epsilon', type=float, default = 1e-8)
    parser.add_argument('--max_grad_norm', type=float, default = 1.0)
    parser.add_argument('--num_train_epochs', '-epoch', type=int, default = 5)
    parser.add_argument('--max_steps', type=int, default = -1)
    parser.add_argument('--warmup_steps', type=int, default = 0)
    parser.add_argument('--logging_steps', '-log', type=int, default = 1000)
    parser.add_argument('--save_steps', '-save', type=int, default = 1000)
    parser.add_argument('--save_total_limit', default = None)
    parser.add_argument('--eval_all_checkpoints', type=bool, default = False)
    parser.add_argument('--no_cuda', type=bool, default = True)
    parser.add_argument('--overwrite_output_dir', type=bool, default = True)
    parser.add_argument('--overwrite_cache', type=bool, default = True)
    parser.add_argument('--should_continue', '-continue', type=bool, default = False)
    parser.add_argument('--seed', type=int, default = 42)
    parser.add_argument('--local_rank', type=int, default = -1)
    parser.add_argument('--fp16', type=bool, default = False)
    parser.add_argument('--fp16_opt_level', type=str, default = 'O1')
    parser.add_argument('--path_train', '-pt', type=str, default = 'data/sample.csv')
    parser.add_argument('--path_validation', '-pv', type=str, default = 'data/validation.csv')

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    
    trn_df = pd.read_csv(args.path_train)
    val_df = pd.read_csv(args.path_validation)
    main(args, logger, trn_df, val_df)