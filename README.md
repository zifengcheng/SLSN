# A Symmetric Local Search Network for Emotion-Cause Pair Extraction

This repo contains the code of our COLING2020 paper:

A Symmetric Local Search Network for Emotion-Cause Pair Extraction

## Requirements

- Python 3.6
- PyTorch 1.4.0

## Training
```shell
python train.py --lamda 0.6 --belta 0.8 --window 5 --lr 0.005
```

## Word embedding
You can use the [url](https://github.com/NUSTM/ECPE/tree/master/data_combine) to download word embedding file w2v_200.txt and put this file in the data folder.

## Citation

If you think the codes & paper are helpful, please cite this paper. Thank you! 

``` bibtex
@inproceedings{Cheng20,
  author    = {Zifeng Cheng and
               Zhiwei Jiang and
               Yafeng Yin and
               Hua Yu and
               Qing Gu},
  title     = {A Symmetric Local Search Network for Emotion-Cause Pair Extraction},
  booktitle = {Proceedings of the 28th International Conference on Computational
               Linguistics, {COLING} 2020, Barcelona, Spain (Online), December 8-13,
               2020},
  pages     = {139--149},
  publisher = {International Committee on Computational Linguistics},
  year      = {2020},
  url       = {https://doi.org/10.18653/v1/2020.coling-main.12},
  doi       = {10.18653/v1/2020.coling-main.12},
}
```


