--model_name_or_path="facebook/wav2vec2-large-xlsr-53-german" \
--num_train_epochs="1" \
--learning_rate="1e-4" \
--weight_decay=0.005 \
--per_device_train_batch_size="12" #pro, kaggle \
--per_device_eval_batch_size="12"  #pro, 8=kaggle \
--report_to="wandb" \
--run_name="colab_full_aug"
--group_by_length \
--save_steps="500" \
--eval_steps="500" #m=5k \
--logging_steps="500" \
--warmup_steps="400" #1000, m=500 \
--do_eval \
--do_train \
--fp16 \
--dataset_config_name="de" \
--preprocessing_num_workers="2" \
--dataloader_num_workers="2" \
--overwrite_output_dir \
--evaluation_strategy="steps" \
--save_total_limit="2" \
--freeze_feature_extractor \
--gradient_checkpointing="1" \
--eval_accumulation_steps="2" \
--dataset_path="/share/datasets/vf-cv_cv_full_fb/" \
--output_dir="/share/w2v/wav2vec2-large-xlsr-german-sm"