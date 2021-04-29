import logging
import os
import sys
import wandb

import transformers
from transformers import (
    Wav2Vec2Processor, Wav2Vec2ForCTC,
)
from transformers.trainer_utils import is_main_process, get_last_checkpoint

from audioengine.model.finetuning.wav2vec2.parquetdataset import ParquetDataset
from audioengine.model.finetuning.wav2vec2.argument_parser import argument_parser
from audioengine.model.finetuning.wav2vec2.wav2vec2_trainer import CustomProgressBarCallback, DataCollatorCTCWithPadding
from audioengine.model.finetuning.wav2vec2.wav2vec2 import load_trainer

logger = logging.getLogger(__name__)


def main():
    model_args, data_args, training_args = argument_parser(sys.argv)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    train_dataset = ParquetDataset(data_args, split='train')
    eval_dataset = ParquetDataset(data_args, split='eval')

    logger.info(f"Split Train[{len(train_dataset)}], Eval[{len(eval_dataset)}]")

    logger.warning(f"Load Processor {training_args.output_dir}")
    # processor = Wav2Vec2Processor.from_pretrained(training_args.output_dir)

    logger.warning(f"Load Model {model_args.model_name_or_path}")
    logger.warning(f"* Cache_Dir {model_args.cache_dir}")

    processor = Wav2Vec2Processor.from_pretrained(model_args.model_name_or_path)

    model = Wav2Vec2ForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        activation_dropout=model_args.activation_dropout,
        attention_dropout=model_args.attention_dropout,
        hidden_dropout=model_args.hidden_dropout,
        feat_proj_dropout=model_args.feat_proj_dropout,
        mask_time_prob=model_args.mask_time_prob,
        gradient_checkpointing=model_args.gradient_checkpointing,
        layerdrop=model_args.layerdrop,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )


    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    logger.info(f"Freeze_Feature_Exctrator: {model_args.freeze_feature_extractor}")

    if model_args.freeze_feature_extractor:
        model.freeze_feature_extractor()

    trainer = load_trainer(model, processor, data_collator, training_args, train_dataset, eval_dataset)

    logger.info("Training-Args:")
    logger.info(training_args)

    results = {}

    if training_args.do_train:
        if "all" in training_args.report_to or "wandb" in training_args.report_to:
            wandb.login()
            lr = "{:.2e}".format(training_args.learning_rate)
            bs = training_args.per_device_train_batch_size
            ep = training_args.num_train_epochs
            training_args.run_name = f"{ep}_{lr}_{bs}"

            logger.info(f"WANDB Run Name: {training_args.run_name}")
        else:
            wandb.init(mode="disabled")

        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        logger.info('Training...')
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        logger.info("train_result:")
        logger.info(train_result)
        results["train_result"] = train_result

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        results["eval_metrics"] = metrics
        logger.info("eval_result:")
        logger.info(results["eval_metrics"])

    args = {
        "model_args": model_args,
        "data_args": data_args,
        "training_args": training_args
    }

    logger.info("args:")
    logger.info(args)

    return results


if __name__ == "__main__":
    print(main())
