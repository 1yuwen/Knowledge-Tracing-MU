# Lifting Data-Tracing Machine Unlearning to Knowledge-Tracing for Foundation Models
<div align="center">

<div>
    <a href='' target='_blank'>Yuwen Tan</a> and <a href='' target='_blank'>Boqing Gong</a>
</div>

<div>
Department of Computer Science, Boston University &emsp;
</div>
</div>

The code repository for "[Lifting Data-Tracing Machine Unlearning to
Knowledge-Tracing for Foundation Models](https://arxiv.org/abs/2403.19979)" in PyTorch. 
    

## News

[06/2025] 🌟 [arXiv](https://arxiv.org/abs/2403.19979) paper has been released.

[06/2025] 🌟 The code repository of the case study has been released.


## Abstract
Machine unlearning removes certain training data points and their influence on AI models (e.g. when a data owner revokes their decision to allow models to learn from the data). In this position paper, we propose to lift data-tracing machine unlearning to knowledge-tracing for foundation models (FMs). We support this position based on practical needs and insights from cognitive studies. Practically, tracing data cannot meet the diverse unlearning requests for FMs, which may be from regulators, enterprise users, product teams, etc., having no access to FMs' massive training data. Instead, it is convenient for these parties to issue an unlearning request about the knowledge or capability FMs (should not) possess. Cognitively, knowledge-tracing unlearning aligns with how the human brain forgets more closely than tracing individual training data points. Finally, we provide a concrete case study about a vision-language FM to illustrate how an unlearner might instantiate the knowledge-tracing machine unlearning paradigm. 
<div align="center">
<img src="assets/teaser.png" width="96%">
</div>


## Case Study
<div align="center">
<img src="assets/case.png" width="93%">
</div>
<p></p>

<div>
Following this work's position, we provide a concrete case study about Contrastive Language-Image Pretraining [CLIP](https://github.com/openai/CLIP) to bridge the position with real-world applications and, in return, explore the position in depth, spanning multiple factors and perspectives. We envision that Oudi Inc., a car manufacturer and an enterprise user of the CLIP model, has retired their O1 sedan for some reason. Accordingly, Oudi's product team requests that the Oudi O1 concept be unlearned from CLIP. An unlearner is equipped with existing MU methods developed in the research community but realizes they all operate on the training data points. The unlearner cannot access CLIP's training data; instead, they assemble a set of exemplar Oudi O1 images as the proxy forgetting set $\mathcal{D}^f$ (but no retention set for convenience).
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
1. [torch 1.12.1](https://github.com/pytorch/pytorch)
2. [torchvision 0.13.1](https://github.com/pytorch/vision)

### Datasets
We provide the processed datasets as follows:
- **ImageNet-1k**: Reference [ImageNet-1k](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)
- **Compcars**: Reference [CompCars](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)

You need to modify the path of the datasets in `./data/.../test.jsonl` and `./data/.../train.jsonl` according to your own path.

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
This repo is based on [CLIP](https://github.com/openai/CLIP) and [PyCIL](https://github.com/G-U-N/PyCIL).

Thanks for their wonderful work!!!

## Correspondence
If you have any question about this project, please contact yuwentan@bu.edu