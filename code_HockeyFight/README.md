# video action recognition experiment on dataset HockeyFight

Video classification using Dataset: HockeyFights

```
code_HockeyFight
├── extract_frames.py
├── hockey_data.py
├── hockey_model_lstm.py
├── hockey_model_3dcnn.py
├── hockey_model_ARTNet.py
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
Download the Dataset [HockeyFight](https://paperswithcode.com/dataset/hockey-fight-detection-dataset) 


To extract frames from the videos, please do:


    ```
    python extract_frames.py
    ```
    
    
To prepare the data for the training and testing and generate hdf5 files, please do:


    ```
    python hockey_data.py
    ```
    
    
(pay attention to change the absolute path in the file train.txt and test.txt )

### Training and Testing


 To train the biLSTM model, please do:
 
 
    ```
    python hockey_model_lstm.py
     ```
     
     
 To train the C3D model, please do:  
 
 
      ```
    python hockey_model_3dcnn.py
     ```
     
     
 To train the ARTNet model, please do:   
 
 
    ```
    python hockey_model_ARTNet.py
    ```


## Results and Models

### HockeyFights

| Model | Input size | acc |
| :---: | :---: | :---: | 
|  C3D  |     16 x 112 x 112     |  93.50  | 
|  biLSTM + Attention  |   16 x 112 x 112     |  95.50  | 
|  ARTNet  |     16 x 112 x 112      |  98.00  |
