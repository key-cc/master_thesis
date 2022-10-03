# Master Thesis: Action Recognition in Video

This repo will serve as a summary of the code in my master thesis. The video action recognition model includes BiLSTM, TSN, I3D, TimeSformer.(For the architecture of TSN, I3D, TimeSformer I used the open-source toolbox [MMaction2](https://github.com/open-mmlab/mmaction2) to help me build the models.)

```
mmaction2
├── mmaction
├── tools
├── configs
│   ├── _base_
│   ├── recognition
│   │   ├── i3d
│   │   │   ├── my_i3d_r50_32x2x1_50e_hmdb51_rgb
│   │   │   ├── my_i3d_r50_32x2x1_50e_ucf101_rgb
│   │   │   ├── ...
│   │   ├── timeSformer
│   │   │   ├── ...
├── data
│   ├── hmdb51
│   │   ├── hmdb51_{train,val}_split_{1,2,3}_rawframes.txt
│   │   ├── hmdb51_{train,val}_split_{1,2,3}_videos.txt
│   │   ├── annotations
│   │   ├── videos
│   │   │   ├── brush_hair
│   │   │   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0.avi
│   │   ├── rawframes
│   │   │   ├── brush_hair
│   │   │   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0
│   │   │   │   │   ├── img_00001.jpg
│   │   │   │   │   ├── img_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   ├── ...
│   ├── ucf101
│   │   ├── ucf101_{train,val}_split_{1,2,3}_rawframes.txt
│   │   ├── ucf101_{train,val}_split_{1,2,3}_videos.txt
│   │   ├── annotations
│   │   ├── videos
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01.avi
│   │   ├── rawframes
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01
│   │   │   │   │   ├── img_00001.jpg
│   │   │   │   │   ├── img_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   ├── ...
│   ├── sthv1
│   │   ├── sthv1_{train,val}_list_rawframes.txt
│   │   ├── sthv1_{train,val}_list_videos.txt
│   │   ├── annotations
│   |   ├── videos
│   |   |   ├── 1.mp4
│   |   |   ├──...
│   |   ├── rawframes
│   |   |   ├── 1
│   |   |   |   ├── 00001.jpg
│   |   |   |   ├── ...
│   |   |   ├── ...

```


I will mainly use the [UCF-101 dataset](https://www.crcv.ucf.edu/data/UCF101.php), [HMDB51 dataset](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) and [Something-Something V1 dataset](https://20bn.com/datasets/something-something/v1).

<p align="center">
    <img src="assets/crawling.gif" width="400"\>
</p>

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
Colaboratory file: [I3D+TimeSformer](hmdb51.ipynb) 


# Action Recognition in Video

This repo will serve as a playground where I investigate different approaches to solving the problem of action recognition in video.

I will mainly use the [UCF-101 dataset](https://www.crcv.ucf.edu/data/UCF101.php).

<p align="center">
    <img src="assets/crawling.gif" width="400"\>
</p>

## Setup

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

I have made a trained model available [here](https://drive.google.com/open?id=1GlpN0m9uLbI9dg1ARbW9hDEf-VWe4Asl).

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

<p align="center">
    <img src="assets/penalty.gif" width="400"\>
</p>

### Results

The model reaches a classification accuracy of **91.27%** accuracy on a randomly sampled test set, composed of 20% of the total amount of video sequences from UCF-101. Will re-train this model on the offical train / test splits and post results as soon as I have time.
