# Master Thesis: Action Recognition in Video

This repo will serve as a summary of the code in my master thesis. The video action recognition model includes ConvLSTM, P3D, ARTNet, Res3D, Res21D. I will mainly use the [UCF-101 dataset](https://www.crcv.ucf.edu/data/UCF101.php), [HMDB51 dataset](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) and [HockeyFight dataset](https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes).

```
master_thesis
├── dataset.py
├── log.py
├── models.py
├── test.py
├── train.py
├── test_on_video.py
├── code_HockeyFight
├── code_UCF101_HMDB51

```

## Dataset UCF101
<!-- [DATASET] -->


```BibTeX
@article{Soomro2012UCF101AD,
  title={UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild},
  author={K. Soomro and A. Zamir and M. Shah},
  journal={ArXiv},
  year={2012},
  volume={abs/1212.0402}
}
```

## Dataset HMDB51
<!-- [DATASET] -->

```BibTeX
@article{Kuehne2011HMDBAL,
  title={HMDB: A large video database for human motion recognition},
  author={Hilde Kuehne and Hueihan Jhuang and E. Garrote and T. Poggio and Thomas Serre},
  journal={2011 International Conference on Computer Vision},
  year={2011},
  pages={2556-2563}
}
```

## Dataset HockeyFights
<!-- [DATASET] -->

```BibTeX
@inproceedings{Nievas2011ViolenceDI,
  title={Violence Detection in Video Using Computer Vision Techniques},
  author={Enrique Bermejo Nievas and Oscar D{\'e}niz-Su{\'a}rez and Gloria Bueno Garc{\'i}a and Rahul Sukthankar},
  booktitle={CAIP},
  year={2011}
}

```

## Setup

```
cd data/              
bash download_ucf101.sh     # Downloads the UCF-101 dataset (~7.2 GB)
unrar x UCF101.rar          # Unrars dataset
unzip UCF101TrainTestSplits-RecognitionTask.zip  # Unzip train / test split
python3 extract_frames.py   # Extracts frames from the video (~26.2 GB, go grab a coffee for this)
```



### Test on Video

```
$ python3 test_on_video.py  --video_path data/UCF-101/SoccerPenalty/v_SoccerPenalty_g01_c01.avi \
                            --checkpoint_model model_checkpoints/ConvLSTM_150.pth
```


### Results
### UCF101

| Model | Parameters | acc |
| :---: | :---: | :---: | 
|  C3D (pretrained) |    78.00M    |  77.58  | 
|  biLSTM + Attention (with dropout) |   74M    | 74.46   | 
|  VTN (pretrained) |    25.54M     |  86.09  |
|Divided Space-Time Attention (T+S) |  121.34M | 93.11|
|Joint Space-Time Attention (ST)|  85.88M|91.83|
|Space Attention Attention (S)|  85.88M | 91.36|
|Swin-T| 49.59M | 92.60|

<div align="left">
  <div style="float:left;margin-right:10px;">
  <img src="https://github.com/key-cc/master_thesis/blob/main/code_UCF101_HMDB51/test_ucfn.gif" width="380px"><br>
    <p style="font-size:1.5vw;">An example of action recognition on UCF101</p>
  </div>

### HMDB51

| Model | Parameters | acc |
| :---: | :---: | :---: | 
|  C3D (pretrained) |     78.20M     |  67.60  | 
|  biLSTM + Attention  |   73.95M    |  62.46  | 
|  VTN (pretrained) |     28.67M      |  60.25 |
|Divided Space-Time Attention (T+S) |  121.3M | 66.08|
|Joint Space-Time Attention (ST)|  85.84M|64.25|
|Space Attention Attention (S)|  85.84M | 65.49|
|Swin-T| 49.55M | 66.25|

<div align="left">
  <div style="float:left;margin-right:10px;">
  <img src="https://github.com/open-mmlab/mmaction2/raw/master/resources/mmaction2_overview.gif" width="380px"><br>
    <p style="font-size:1.5vw;">An example of action recognition on HMDB51</p>
  </div>

### HockeyFights

| Model | Parameters| acc |
| :---: | :---: | :---: | 
|  C3D  |     78M    |  93.50  | 
|  biLSTM + Attention  |  26.41M     |  95.50  | 
|  ARTNet  |    20.15M     |  98.00  |
|Divided Space-Time Attention (T+S)| 121.27M| 93.50|
|Joint Space-Time Attention (ST)| 85.81M| 92.00|
|Space Attention Attention (S)|  85.80M |91.50|

<div align="left">
  <div style="float:left;margin-right:10px;">
  <img src="https://github.com/key-cc/master_thesis/blob/main/code_HockeyFight/test_hockeyfight.gif" width="380px"><br>
    <p style="font-size:1.5vw;">An example of action recognition on HockeyFights</p>
  </div>
