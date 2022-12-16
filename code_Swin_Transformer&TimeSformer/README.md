# Video Swin Transformer & TimeSformer on Dataset UCF101

## Usage
Download the documents from [open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2). Initialize the environment and download the needed packages following the instruction. Replace the folder 'configs' with the new one here. Compared with the original folder, the new one has three added files in the folder timesformer and three in the folder swin. All the files use the dataset UCF101.

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

## Citation

```
@article{liu2021video,
  title={Video Swin Transformer},
  author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  journal={arXiv preprint arXiv:2106.13230},
  year={2021}
}

@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```
```BibTeX
@misc{bertasius2021spacetime,
    title   = {Is Space-Time Attention All You Need for Video Understanding?},
    author  = {Gedas Bertasius and Heng Wang and Lorenzo Torresani},
    year    = {2021},
    eprint  = {2102.05095},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
The code is adapted from the official Video-Swin-Transformer repository[open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2). Thanks for their sharing.
