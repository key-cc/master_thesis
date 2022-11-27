# video action recognition experiment on dataset HockeyFight

Video classification using Dataset: HockeyFights
```
code_HockeyFight
├── extract_frames.py
├── hockey_data_lstm.py
├── hockey_model_lstm.py
├── test.txt
├── train.txt
├── log.py
├── data
│   ├── fi100_xvid
│   ├── fi101_xvid
│   ├── fi102_xvid
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├──  ...
│   ├── ...
├── video
│   ├── HockeyFights
│   │   ├── fi100_xvid.avi
│   │   ├── fi101_xvid.avi
│   │   ├── fi102_xvid.avi
│   │   ├──  ...

```
## Setup
###  Data Preparation


### Training

### Testing

## Results and Models

### HockeyFights

| Model | Input size | acc |
| :---: | :---: | :---: | 
|  C3D  |     16 x 112 x 112     |  93.50  | 
|  biLSTM + Attention  |   16 x 112 x 112     |  95.50  | 
|  ARTNet  |     16 x 112 x 112      |  98.00  |
