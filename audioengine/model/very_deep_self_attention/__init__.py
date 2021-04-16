#from audioengine.model.very_deep_self_attention.helper import get_longest_string, get_data_from_df, create_tf_dataset, \
    #train_model
#from audioengine.model.very_deep_self_attention.summed import *
#from audioengine.corpus.dataset import Dataset, CommonVoice

#common_voice_path = "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de"
#audio_format = "mp3"
#epochs = 1
#batch_size = 32

#common_voice = CommonVoice(common_voice_path)
#train_df = common_voice.load_dataframe(type="train")  # train
#test_df = common_voice.load_dataframe(type="test")
#input_key, output_key = "path", "sentence"

## max_target_len_train = get_longest_string(train_df, output_key)
## max_target_len_test = get_longest_string(test_df, output_key)
## max_target_len = max(max_target_len_train, max_target_len_test)

## print(max_target_len) #157
#max_target_len = 157

#train_data = get_data_from_df(train_df, input_key, output_key)
#test_data = get_data_from_df(test_df, input_key, output_key)

#train_model(train_data,test_data,max_target_len, audio_format, batch_size, epochs)
