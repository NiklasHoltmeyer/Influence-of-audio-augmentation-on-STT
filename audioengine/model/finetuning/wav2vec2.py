import transformers

from audioengine.model.finetuning.helper.finetune import CustomProgressBarCallback
from audioengine.model.pretrained.wav2vec2 import wav2vec2
from transformers import TrainingArguments, Trainer

from audioengine.model.finetuning.helper.wav2vec2 import compute_metrics, GroupedLengthsTrainer


def load_training_arguments_from_kwargs(output_dir, **kwargs):
    return TrainingArguments(
        output_dir=output_dir,
        group_by_length=kwargs.get("group_by_length", True),
        per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=kwargs.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 2),
        evaluation_strategy=kwargs.get("evaluation_strategy", "steps"),
        num_train_epochs=kwargs.get("num_train_epochs", 30),
        fp16=kwargs.get("fp16", True),
        save_steps=kwargs.get("save_steps", 400),
        eval_steps=kwargs.get("eval_steps", 400),
        logging_steps=kwargs.get("logging_steps", 400),
        logging_dir=kwargs.get("logging_dir", "./log/"),
        learning_rate=kwargs.get("learning_rate", 3e-4),
        weight_decay=kwargs.get("weight_decay", 0.005),
        warmup_steps=kwargs.get("warmup_steps", 500),
        save_total_limit=kwargs.get("save_total_limit", 2),
        dataloader_num_workers=kwargs.get("dataloader_num_workers", 0)
    )


def load_trainer(model, processor, data_collator, args, train_dataset=None, eval_dataset=None, **kwargs):
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=args,
        compute_metrics=kwargs.get("compute_metrics", compute_metrics(processor)),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)
    trainer.add_callback(CustomProgressBarCallback)
    return trainer


def load_grouped_trainer(model, processor, data_collator, args, train_dataset=None, eval_dataset=None, **kwargs):
    trainer = GroupedLengthsTrainer(
        model=model,
        data_collator=data_collator,
        args=args,
        compute_metrics=kwargs.get("compute_metrics", compute_metrics(processor)),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
        train_seq_lengths=train_dataset.input_seq_lengths
    )
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)
    trainer.add_callback(CustomProgressBarCallback)
    return trainer

# wer_metric = load_metric("wer")

# def compute_metrics(pred):
#    pred_logits = pred.predictions
#    pred_ids = np.argmax(pred_logits, axis=-1)

#    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

#    pred_str = processor.batch_decode(pred_ids)
#    # we do not want to group tokens when computing the metrics
#    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

#    wer = wer_metric.compute(predictions=pred_str, references=label_str)

#    return {"wer": wer}
