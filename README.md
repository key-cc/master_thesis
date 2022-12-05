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

| Model | Input size | acc |
| :---: | :---: | :---: | 
|  C3D  |     16 x 224 x 224     |  51.10  | 
|  C3D (pretrained) |     16 x 224 x 224     |  77.58  | 
|  biLSTM + Attention  |   16 x 224 x 224     |  73.20  | 
|  biLSTM + Attention (with dropout) |   16 x 224 x 224     | 74.46   | 
|  VTN  |     16 x 224 x 224      |  75.10  |
|  VTN (pretrained) |     16 x 224 x 224      |  86.09  |

### HockeyFights

| Model | Input size | acc |
| :---: | :---: | :---: | 
|  C3D  |     16 x 112 x 112     |  93.50  | 
|  biLSTM + Attention  |   16 x 112 x 112     |  95.50  | 
|  ARTNet  |     16 x 112 x 112      |  98.00  |
