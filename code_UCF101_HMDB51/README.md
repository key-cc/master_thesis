# video action recognition experiment on dataset UCF101

Video classification using Dataset: UCF101

```
code_UCF101
├── dataset.py
├── c3d_dataset.py
├── model_lstm.py
├── c3d_models.py
├── c3d_train.py
├── train_lstm.py
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
├── vtn
│   ├── action_recognition
│   ├── dataset
│   ├── main.py
│   ├── ...

```


## Setup
###  Data Preparation
Download the Dataset ## Dataset UCF101
```
cd master_thesis/code_UCF101/data             
bash download_ucf101.sh     # Downloads the UCF-101 dataset (~7.2 GB)
unrar x UCF101.rar          # Unrars dataset
unzip UCF101TrainTestSplits-RecognitionTask.zip  # Unzip train / test split
```

To extract frames from the videos, please do:

    python extract_frames.py  # Extracts frames from the video (~26.2 GB)
  
    
## ConvLSTM
The model is composed of:
* A convolutional feature extractor (ResNet-152) which provides a latent representation of video frames
* A bi-directional LSTM classifier which based on the latent representation of the video predicts the activity depicted

### Train  

```
$ python3 train_lstm.py  --dataset_path data/UCF-101-frames/ \
                    --split_path data/ucfTrainTestlist \
                    --num_epochs 200 \
                    --sequence_length 40 \
                    --img_dim 112 \
                    --latent_dim 512
```

### Test on Video

```
$ python3 test_on_video.py  --video_path data/UCF-101/SoccerPenalty/v_SoccerPenalty_g01_c01.avi \
                            --checkpoint_model model_checkpoints/ConvLSTM_100.pth
```

## C3D
### Train  

```
$ python3 c3d_train.py  

```
## VTN
```
cd vtn 
```
initialize the enviroment:

```bash
pip install -r requirements.txt
```

install FFmpeg in order to prepare training data:

```bash
sudo apt-get install ffmpeg
```
### Train  
#### Train a Model (with ImageNet Pretrain)

```bash
python3 main.py --root-path ~/data --result-path ~/logs/experiment_name --dataset ucf101 --model resnet34_vtn \
    --batch 64 -j 12 --clip-size 16 --st 2 --n-epochs 120 --lr 1e-4
```    
#### Fine-tune a Pretrained Model
```
python3 main.py --root-path ~/data --result-path ~/logs/ --dataset ucf101 --model resnet34_vtn \
    --batch 64 -j 12 --clip-size 16 --st 2 --lr 1e-5 --pretrain-path ~/resnet34_vtn_kinetcs.pth
```


## Results and Models

### UCF101

| Model | Input size | acc |
| :---: | :---: | :---: | 
|  C3D  |     16 x 224 x 224     |  51.10  | 
|  C3D (pretrained) |     16 x 224 x 224     |  77.58  | 
|  biLSTM + Attention  |   16 x 224 x 224     |  73.20  | 
|  biLSTM + Attention (with dropout) |   16 x 224 x 224     |  74.46  | 
|  VTN  |     16 x 224 x 224      |  75.10  |
|  VTN (pretrained) |     16 x 224 x 224      |  86.09  |

<div align="left">
  <div style="float:left;margin-right:10px;">
  <img src="https://github.com/key-cc/master_thesis/blob/main/code_UCF101_HMDB51/test1_ucf101.gif" width="380px"><br>
    <p style="font-size:1.5vw;">An example of action recognition on UCF101</p>
  </div>
  
  
<div align="left">
  <div style="float:left;margin-right:10px;">
  <img src="https://github.com/key-cc/master_thesis/blob/main/code_UCF101_HMDB51/test_hmdb51.gif" width="380px"><br>
    <p style="font-size:1.5vw;">An example of action recognition on HMDB51</p>
  </div>

