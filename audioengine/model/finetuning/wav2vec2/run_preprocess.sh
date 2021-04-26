python3 preprocess_dataset.py --model_name_or_path="maxidl/wav2vec2-large-xlsr-german" \
--dataset_config_name="de" \
--output_dir=/share/train/w2v/ \
--preprocessing_num_workers="8" \
--overwrite_output_dir \
--num_train_epochs="1" \
--per_device_train_batch_size="32" \
--per_device_eval_batch_size="32" \
--learning_rate="1e-4" \
--warmup_steps="500" \
--evaluation_strategy="steps" \
--save_steps="5000" \
--eval_steps="5000" \
--logging_steps="1000" \
--save_total_limit="3" \
--freeze_feature_extractor \
--activation_dropout="0.055" \
--attention_dropout="0.094" \
--feat_proj_dropout="0.04" \
--layerdrop="0.04" \
--mask_time_prob="0.08" \
--gradient_checkpointing="1" \
--fp16 \
--do_train \
--do_eval \
--dataloader_num_workers="8" \
--group_by_length
#.max_train_samples
#max_val_samples