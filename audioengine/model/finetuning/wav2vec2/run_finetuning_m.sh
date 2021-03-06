--model_name_or_path="maxidl/wav2vec2-large-xlsr-german"
--dataset_config_name="de"
--output_dir="/share/w2v/wav2vec2-large-xlsr-german-sm"
--preprocessing_num_workers="8"
--overwrite_output_dir
--num_train_epochs="1"
--per_device_train_batch_size="4"
--per_device_eval_batch_size="4"
--learning_rate="1e-6"
--warmup_steps="0"
--evaluation_strategy="steps"
--save_steps="400"
--eval_steps="400"
--logging_steps="400"
--save_total_limit="3"
--freeze_feature_extractor
--activation_dropout="0.055"
--attention_dropout="0.094"
--feat_proj_dropout="0.04"
--layerdrop="0.04"
--mask_time_prob="0.08"
--gradient_checkpointing="1"
--eval_accumulation_steps="2"
--fp16
--do_eval
--do_train
--dataloader_num_workers="8"
--group_by_length
--preprocess_dataset_train_path="/share/datasets/vf_de/"
--preprocess_dataset_eval_path="/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de"
--dataset_path="/share/datasets/vf80full_cv20-test_de_processed/"