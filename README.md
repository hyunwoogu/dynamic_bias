# Attractor dynamics of working memory explain a concurrent evolution of stimulus-specific and decision-consistent biases in visual estimation

This repository hosts the source code for the [Preprint](https://www.biorxiv.org/content/10.1101/2023.06.28.546818v1).

## Setup instructions

### Installation

The source code has been tested under macOS 15.0.1 without dependence on GPU. To install the project package, run the following.
```bash
git clone https://github.com/hyunwoogu/dynamic_bias
cd dynamic_bias
pip install -e .
```

### Generating outputs

To generate the analysis outputs, run the following. This includes the model fits.
```bash
make analysis-behavior
make analysis-ddm
make analysis-fmri
make analysis-rnn
```

### Data and model weights

To skip the previous part and directly obtain the outputs and the RNN model weights, download the following OSF link. Place them under `dynamic_bias/data/` and `dynamic_bias/models/`, respectively.

[Data and model weights](https://osf.io/6q95m/)


### Notebooks

To replicate the figures in the paper, run the jupyter notebooks under `notebooks/`. Note that as these notebooks contain the code for downloading the required output files, the previous steps of generating outputs or training models may be skipped. Expected run time of each notebook is less than a minute.

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