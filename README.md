# Decision-consistent bias mediated by drift dynamics of human visual working memory

This repository hosts the code, data, and model weights for our paper. 

[Preprint](https://www.biorxiv.org/content/10.1101/2023.06.28.546818v1)


## Setup instructions

### Environment

To set up a virtual environment and install dependencies, run the following.
```
python -m venv env
source env/bin/activate
make init
```

### Generating outputs

To generate the analysis outputs, run the following.
```
make analysis-behavior
make analysis-fmri
make model-ddm
make model-rnn
```

### Notebooks

To replicate the figures in the paper, visit `notebooks/` and run the corresponding jupyter notebook. As these notebooks contain the code for downloading the required output files, the previous step of generating outputs may be skipped.


## Citation

```
@article{gu2023decision,
  title={Decision-consistent bias mediated by drift dynamics of human visual working memory},
  author={Gu, Hyunwoo and Lee, Joonwon and Kim, Sungje and Lim, Jaeseob and Lee, Hyang-Jung and Lee, Heeseung and Choe, Minjin and Yoo, Dong-Gyu and Ryu, Jun Hwan (Joshua) and Lim, Sukbin and Lee, Sang-Hun},
  journal={bioRxiv},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```