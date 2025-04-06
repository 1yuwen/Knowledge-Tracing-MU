# Lifting Data-Tracing Machine Unlearning to Knowledge-Tracing for Foundation Model Unlearning
<div align="center">

<div>
    <a href='' target='_blank'>Yuwen Tan</a><sup>1</sup>&emsp;
    <a href='' target='_blank'>Boqing Gong</a><sup>1</sup>&emsp;
    
</div>
<div>
<sup>1</sup>Department of Computer Science, Boston University &emsp;
</div>
</div>

The code repository for "[Lifting Data-Tracing Machine Unlearning to Knowledge-Tracing for Foundation Model Unlearning](https://arxiv.org/abs/2403.19979)" in PyTorch. 
    

## News

[03/2025] 🌟 [arXiv](https://arxiv.org/abs/2403.19979) paper has been released.

[04/2025] 🌟 The code repository  has been released.


## Abstract
Machine unlearning removes certain training data points and their influence on AI models (e.g. when a data owner revokes their decision to allow models to learn from the data). In this position paper, we propose to lift data-tracing machine unlearning to knowledge-tracing for foundation models (FMs). We support this position based on practical needs and insights from cognitive studies. Practically, tracing data cannot meet the diverse unlearning requests for FMs, which may be from regulators, enterprise users, product teams, etc., having no access to FMs' massive training data. Instead, it is convenient for these parties to issue an unlearning request about the knowledge or capability FMs (should not) possess. Cognitively, knowledge-tracing unlearning aligns with how the human brain forgets more closely than tracing individual training data points. Finally, we provide a case study using a vision-language foundation model to deepen the discussions.
<p></p>

## Pipeline
<div align="center">
<img src="results/overall_process.png" width="93%">
</div>
<p></p>

<div>
We use adapter without parameter limitation as our baseline, compared with other PETuning method, we find adapter performs best in balancing the performance of old and new classes. We further train the classifier by sampling features with Gaussian samples, which improves the performance of the incremental process. During the construction of the distribution, we apply semantic bias correction to the prototype of each feature within each class.
</div>

<p></p>

## Results
<div>
The following table shows the main results of our proposed method and other SOTA methods. Please note that there might be slight variations in results based on the type and quantity of NVIDIA GPUs.
</div>

<div align="center">
<img src="results/main table.JPG" width="96%">
</div>


## Requirements
### Dependencies
1. [torch 1.11.0](https://github.com/pytorch/pytorch)
2. [torchvision 0.12.0](https://github.com/pytorch/vision)
3. [timm 0.6.12](https://github.com/huggingface/pytorch-image-models)


### Datasets
We provide the processed datasets as follows:
- **CIFAR100**: will be automatically downloaded by the code.
- **CUB200** **ImageNet-R** **ImageNet-A**  **VTAB**: Reference [Revisiting](https://github.com/zhoudw-zdw/RevisitingCIL)


These subsets are sampled from the original datasets. Please note that we  do not have the right to distribute these datasets. If the distribution violates the license, I shall provide the filenames instead.

You need to modify the path of the datasets in `./data/data.py`  according to your own path.

## Training Scripts
Please follow the settings in the `exps` folder to prepare your json files, and then run:

```
python main.py --config ./exps/[configname].json

for imageneta:
python main.py --config ./exps/adapter_imageneta.json
for imagenetr:
python main.py --config ./exps/adapter_imagenetr.json
for cifar224:
python main.py --config ./exps/adapter_cifar224.json
for cub200:
python main.py --config ./exps/adapter_cub.json

```


## Citation
If you find this useful in your research, please consider citing:
```
@article{tan2024semantically,
 title={Semantically-Shifted Incremental Adapter-Tuning is A Continual ViTransformer},
 author={Tan, Yuwen and Zhou, Qinhao and Xiang, Xiang and Wang, Ke and Wu, Yuchuan and Li, Yongbin},
 journal={arXiv preprint arXiv:2403.19979},
 year={2024}
}

```

## Acknowledgment
This repo is based on [RevisitingCIL](https://github.com/zhoudw-zdw/RevisitingCIL) and [PyCIL](https://github.com/G-U-N/PyCIL).

The implemenations of parameter-efficient tuning methods are based on [VPT](https://github.com/sagizty/VPT), [AdaptFormer](https://github.com/ShoufaChen/AdaptFormer), and [SSF](https://github.com/dongzelian/SSF).

Thanks for their wonderful work!!!

## Correspondence
If you have any question about this project, please contact xex@hust.edu.cn