# FedPHA: Federated Prompt Learning for Heterogeneous Client Adaptation [ICML 2025]
The implementation of paper **FedPHA: Federated Prompt Learning for Heterogeneous Client Adaptation (ICML 2025)**.
[[paper]](https://openreview.net/forum?id=y7pDvbi9xz)
![FedPHA-pipeline](FedPHA-pipeline.jpg "FedPHA-pipeline")

## Requirements
- Python 3.8+
- Pytorch 1.10.0+

To install requirements:
```
pip install -r requirements.txt
```

## Data Preparation
Please follow the instructions at [CoOP](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to prepare the following datasets: Caltech101, OxfordPets, Flowers102, Food101, DTD.

For CIFAR10 and CIFAR100 datasets, please download and unzip data under `DATA/` file catalog. Or simply run experiments with CIFAR10/CIFAR100 dataset, the program will download data automatically.

For DomainNet and office-caltech10 datasets, please follow the instructions of Dataset described [here](https://github.com/med-air/FedBN/blob/master/README.md). 

## How to Run

You can run `federated_main.py` with some specified arguments.

### Training

`--root` takes as input a path to dataset, like `caltech101` or `oxford_flowers`.

`--config-file` means which config file to use, such as `rn50` or `vit_b16`.

You can select variables like shots, clients by changing `cfg` or you can change every arguments you like in `FedPHA_few_shot.sh`.

### For example
If you want to use FedPHA to train caltech101 dataset with 2 shots, backbone rn50 and total independent non-iid setting.
You can specify that:
```
TRAINER=GL_SVDMSE
DATA=caltech101
SHOTS=2
USEALL=False
IID=False
```
and run `bash scripts/FedPHA_few_shot.sh`

After the experiments, all the results are finished and save to `output/`.

### Heterogeneous Scenario Experiments
We further provide a script to evaluate FedPHA under heterogeneous prompt-length settings across clients.  

The script `scripts/best_prompts_list.sh` automatically sweeps different prompt length combinations (e.g., [4, 8, 12, …, 32]) for multiple domains, dispatches tasks across GPUs, and saves results into `output/`. 

Run it with`bash scripts/FedPHA_HE_prompts.sh`


## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{fangfedpha2025,
  title={FedPHA: Federated Prompt Learning for Heterogeneous Client Adaptation},
  author={Fang, Chengying and Huang, Wenke and Wan, Guancheng and Yang, Yihao and Ye, Mang},
  booktitle={Forty-second International Conference on Machine Learning}
  year={2025}
}
```
