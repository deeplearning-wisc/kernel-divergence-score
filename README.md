# Platonic Alignment

Official implementation of ICML 2025 paper, "[How Contaminated Is Your Benchmark? Measuring Dataset Leakage in Large Language Models with Kernel Divergence](https://arxiv.org/abs/2502.00678)" by Hyeong Kyu Choi*, Maxim Khanov*, Hongxin Wei, and Yixuan Li.

## Setup Environment
```
git clone https://github.com/deeplearning-wisc/kernel-divergence-score.git
cd kernel-divergence-score
```

```
conda env create -f environment.yml
conda activate kds
```


## Experiments

Experiment commands are in ```scripts/```. Each shell file computes the kernel divergence scores for contamination rate 0.0~1.0 on seed 0.

```
sh scripts/wikimia.sh
```


```
sh scripts/bookmia.sh
```

```
sh scripts/arxivtection.sh
```

```
sh scripts/pile.sh
```


## Citation
```
@inproceedings{choi2024beyond,
      title={How Contaminated Is Your Benchmark? Measuring Dataset Leakage in Large Language Models with Kernel Divergence}, 
      author={Hyeong Kyu Choi and Maxim Khanov and Hongxin Wei and Yixuan Li},
      booktitle = {International Conference on Machine Learning},
      year = {2025}
}
```