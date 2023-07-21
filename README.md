# Decision-consistent bias mediated by drift dynamics of human visual working memory

This repository hosts the source code for our paper.

[Preprint](https://www.biorxiv.org/content/10.1101/2023.06.28.546818v1)


## Setup instructions

### Environment

The source code has been tested under macOS 13.4.1 without dependence on GPU. To set up a virtual environment and install dependencies, run the following. Specific versions of the dependencies are listed in `requirements.txt`. Expected install time is less than ten minutes.
```
python -m venv env
source env/bin/activate
make init
```

### Generating outputs

To train the models and generate the analysis outputs, run the following.
```
make analysis-behavior
make analysis-fmri
make model-ddm
make model-rnn
```

### Data and model weights

To skip the previous part and directly obtain the outputs and model weights, download the following OSF link. Place them under `dynamic_bias/data/` and `dynamic_bias/models/`, respectively.

[Data and model weights](https://osf.io/6q95m/)


### Notebooks

To replicate the figures in the paper, visit `notebooks/` and run the corresponding jupyter notebook. We believe each notebook is self-explanatory. Note that as these notebooks contain the code for downloading the required output files, the previous steps of generating outputs or training models may be skipped. Expected run time of each notebook is less than three minutes.


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