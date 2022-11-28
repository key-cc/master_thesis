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
├── data
│   ├── _init_.py
│   ├── download_ucf101.sh
│   ├── extract_frames.py
│   ├── UCF-101
│   │   ├── ApplyEyeMakeup
│   │   │   ├── v_ApplyEyeMakeup_g01_c01.avi
│   │   │   ├── ...
│   │   ├── ...
│   ├── UCF-101-frames
│   │   ├── ApplyEyeMakeup
│   │   │   ├── v_ApplyEyeMakeup_g01_c01
│   │   │   │   ├── 0.jpg
│   │   │   │   ├── 1.jpg
│   │   │   │   ├── ...
│   ├── ucfTrainTestlist
│   │   ├── classlnd.txt
│   │   ├── testlist01.txt
│   │   ├── ...
│   │   ├── trainlist01.txt
│   │   ├── ...

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

Colaboratory file: [I3D+TimeSformer](hmdb51.ipynb) 

```
cd data/              
bash download_ucf101.sh     # Downloads the UCF-101 dataset (~7.2 GB)
unrar x UCF101.rar          # Unrars dataset
unzip UCF101TrainTestSplits-RecognitionTask.zip  # Unzip train / test split
python3 extract_frames.py   # Extracts frames from the video (~26.2 GB, go grab a coffee for this)
```

## ConvLSTM

The only approach investigated so far. Enables action recognition in video by a bi-directional LSTM operating on frame embeddings extracted by a pre-trained ResNet-152 (ImageNet).

The model is composed of:
* A convolutional feature extractor (ResNet-152) which provides a latent representation of video frames
* A bi-directional LSTM classifier which based on the latent representation of the video predicts the activity depicted

### Train  

```
$ python3 train.py  --dataset_path data/UCF-101-frames/ \
                    --split_path data/ucfTrainTestlist \
                    --num_epochs 200 \
                    --sequence_length 40 \
                    --img_dim 112 \
                    --latent_dim 512
```

### Test on Video

```
$ python3 test_on_video.py  --video_path data/UCF-101/SoccerPenalty/v_SoccerPenalty_g01_c01.avi \
                            --checkpoint_model model_checkpoints/ConvLSTM_150.pth
```


### Results

The model reaches a classification accuracy of **91.27%** accuracy on a randomly sampled test set, composed of 20% of the total amount of video sequences from UCF-101. Will re-train this model on the offical train / test splits and post results as soon as I have time.
