# Influence of audio augmentation techniques on STT systems.
This project investigates the impact of audio augmentation techniques on low-resource languages. For this purpose, various models were trained and compared using the same training data, but with different augmentation techniques. The Common Voice dataset was used as training data, which was artificially reduced to 41 hours in order to simulate a low-resource language.

## Augmentation comparison
The following table shows the influence of different augmentations on the training process and thus on the resulting model. [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) was used as the pretrained model, which was further trained on 41 hours of the CommonVoice dataset, which was doubled by augmentation. In the following table, the CV-MD, CV-SM entries describe reference values that were trained without augmentation. CV-SM describes the used training data set without augmentation and CV-MD a data set with two times the amount of data as CV-SM but without the use of augmentation. Thus, the training data sets are exactly the same size for all entries except CV-SM. The CV-MD entry is supposed to represent a best case and CV-SM the reference value without augmentation. 
<p align="center">
<img src="https://raw.githubusercontent.com/NiklasHoltmeyer/stt-audioengine/master/misc/comparison/svg/3_comparison.svg"/>
</p>

The two reference values (CV-SM, -MD) and the best augmentation technique will be evaluated relative to other models in the following. It should be noted that the other models were trained on the complete CV data set, unless otherwise described. The models trained on CV-Train were thus trained on about 9 times as much data.

<p align="center">
<img src="https://raw.githubusercontent.com/NiklasHoltmeyer/stt-audioengine/master/misc/comparison/svg/3_overall.svg"/>
</p>

## Other - comparison of STT solutions
### Datasets
<p align="center">
<img src="https://raw.githubusercontent.com/NiklasHoltmeyer/stt-audioengine/master/misc/comparison/svg/1_datasets.svg"/>
</p>

### Traditional approaches
<p align="center">
<img src="https://raw.githubusercontent.com/NiklasHoltmeyer/stt-audioengine/master/misc/comparison/svg/2_related_work_trad.svg"/>
</p>

### End-to-End approaches
<p align="center">
<img src="https://raw.githubusercontent.com/NiklasHoltmeyer/stt-audioengine/master/misc/comparison/svg/2_related_work_end_to_end_comparison.svg"/>
</p>

## References
<p align="center">
<img src="https://raw.githubusercontent.com/NiklasHoltmeyer/stt-audioengine/master/misc/comparison/svg/references-1.svg"/>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/NiklasHoltmeyer/stt-audioengine/master/misc/comparison/svg/references-2.svg"/>
</p>
