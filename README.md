# Self-Supervised Video Transformer

[Kanchana Ranasinghe](https://kahnchana.github.io),
[Muzammal Naseer](https://muzammal-naseer.netlify.app/),
[Salman Khan](https://salman-h-khan.github.io),
[Fahad Shahbaz Khan](https://sites.google.com/view/fahadkhans/home),
[Michael Ryoo](http://michaelryoo.com)

**[Paper Link](https://arxiv.org/abs/2112.01514)** | **[Project Page](https://kahnchana.github.io/svt)** 


> **Abstract:**
>*In this paper, we propose self-supervised training for video transformers using unlabelled video data. From a given video, we create local and global spatiotemporal views with varying spatial sizes and frame rates. Our self-supervised objective seeks to match the features of these different views representing the same video, to be invariant to spatiotemporal variations in actions. To the best of our knowledge, the proposed approach is the first to alleviate the dependency on negative samples or dedicated memory banks in Self-supervised Video Transformer (SVT). Further, owing to the flexibility of Transformer models, SVT supports slow-fast video processing within a single architecture using dynamically adjusted positional encodings and supports long-term relationship modeling along spatiotemporal dimensions. Our approach performs well on four action recognition benchmarks (Kinetics-400, UCF-101, HMDB-51, and SSv2) and converges faster with small batch sizes.*


## Usage & Data
Refer to `requirements.txt` for installing all python dependencies. We use python 3.7 with pytorch 1.7.1. 

We download the official version of Kinetics-400 from [here](https://github.com/cvdfoundation/kinetics-dataset) and videos are resized using code [here](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics).


## Self-supervised Training
For self-supervised pre-training on models on the Kinetics-400 dataset, use the scripts in the `scripts` directory as follows. Change the paths to dataset as required. 

```
./scripts/train.sh
``` 


## Downstream Evaluation
Scripts to perform evaluation (linear or knn) on selected downstream tasks are as below. Paths to datasets and pre-trained models must be set appropriately. Note that in the case of linear evaluation, a linear layer will be fine-tuned on the new dataset and this training can be time-consuming on a single GPU.  

```
./scripts/eval_linear.sh
./scripts/eval_knn.sh
``` 


## Pretrained Models
Our pre-trained models can be found under [releases](https://github.com/kahnchana/svt/releases/tag/v1.0).


## Citation

```bibtex
@misc{ranasinghe2021selfsupervised,
      title={Self-supervised Video Transformer}, 
      author={Kanchana Ranasinghe and Muzammal Naseer and Salman Khan and Fahad Shahbaz Khan and Michael Ryoo},
      year={2021},
      eprint={2112.01514},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Acknowledgements
Our code borrows heavily from [DINO](https://github.com/facebookresearch/dino) and [TimeSformer](https://github.com/facebookresearch/TimeSformer) repositories. We thank the authors for releasing their code. If you use our model, please consider citing these works as well.
