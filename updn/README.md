## Debiasing the BottomUpTopDown Model for Visual Question Answering
This repo contains code to run the VQA-CP experiments from our paper ["Don’t Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases"](https://arxiv.org/abs/1909.03683).
In particular, it contains code to a train VQA model so that it does
not make use of question-type priors when answering questions, and evaluate it on [VQA-CP v2.0](https://www.cc.gatech.edu/~aagrawal307/vqa-cp/).

This repo is a fork of [this](https://github.com/hengyuan-hu/bottom-up-attention-vqa/) 
implementation of the [BottomUpTopDown VQA model](https://arxiv.org/abs/1707.07998). This fork extends the implementation so it can be used
on [VQA-CP v2.0](https://www.cc.gatech.edu/~aagrawal307/vqa-cp/), and supports the debiasing methods from our paper. 


### Prerequisites

Make sure you are on a machine with a NVIDIA GPU and Python 2 with about 100 GB disk space.

1. Install [PyTorch v0.3](http://pytorch.org/) with CUDA and Python 2.7.
2. Install h5py, pillow, and tqdm           

### Data Setup

All data should be downloaded to a 'data/' directory in the root
directory of this repository.

The easiest way to download the data is to run the provided script
`tools/download.sh` from the repository root. The features are
provided by and downloaded from the original authors'
[repo](https://github.com/peteanderson80/bottom-up-attention). If the
script does not work, it should be easy to examine the script and
modify the steps outlined in it according to your needs. Then run
`tools/process.sh` from the repository root to process the data to the
correct format.

### Setup Example
On a fresh machine with Ubuntu 18.04, I was able to setup everything by installing [Cuda 10.0](https://developer.nvidia.com/cuda-10.0-download-archive), then running:

```
sudo apt update
sudo apt install unzip
sudo apt install python-pip
pip2 install torch==0.3.1
pip2 install h5py tqdm pillow 
bash tools/download.sh
bash tools/process.sh
```

### Training

Run `python main.py --output_dir /path/to/output --seed 0` to start training our Learned-Mixin +H VQA-CP model, see the command line options
for how to use other ensemble method, or how to train on non-changing priors VQA 2.0.

### Testing
The scores reported by the script are very close (within a hundredth of a percent in my experience) to the results
reported by the official evaluation metric, but can be slightly different because the 
answer normalization process of the official script is not fully accounted for.
To get the official numbers, you can run `python save_predictions.py /path/to/model /path/to/output_file`
and the run the official VQA 2.0 evaluation [script](https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvalDemo.py)
on the resulting file.

### Results by Answer Type
We present a breakdown of accuracy by answer type below. The overall accuracies do not precisely 
match the results in the paper because, due to a checkpointing issue, we had to re-run our experiments to
get these numbers. The results are still averaged over eight runs, and are very close to the numbers in the 
paper. 

| Debiasing Method | Overall | Yes/No | Number | Other |
| --- | --- | --- | --- | --- |
|None|39.337|42.134|12.293|45.291|
|Reweight|39.915|44.307|12.521|45.130|
|Bias Product|40.043|43.395|12.322|45.892|
|Learned-Mixin|48.778|72.780|14.608|45.576|
|Learned-Mixin +H|52.013|72.580|31.117|46.968|

### VQA 2.0 Results
We present scores for our methods on VQA 2.0, these were collected by re-training 
the models on the VQA 2.0 train set and testing on the validation set. 
Results are again averaged over eight runs.

| Debiasing Method | Overall | Yes/No | Number | Other |
| --- | --- | --- | --- | --- |
|None|63.377|81.170|42.501|55.373|
|Reweight|62.409|79.506|41.835|54.857|
|Bias Product|63.207|81.016|42.302|55.199|
|Learned-Mixin|63.260|81.159|42.215|55.221|
|Learned-Mixin +H|56.345|65.057|37.631|54.687|

### Code Changes
In general we have tried to minimizes changes to the original codebase to reduce the risk of adding bugs, the main changes are:

1. The download and preprocessing script also setup [VQA-CP 2.0](https://www.cc.gatech.edu/~aagrawal307/vqa-cp/)
2. We use the filesystem, instead of HDF5, to store image feature. On my machine this is about a 1.5-3.0x speed up
3. Support dynamically loading the image features from disk during training so models can be trained
on machines with less RAM
4. Debiasing objectives are added in `vqa_debiasing_objectives.py`
5. Some additional arguments are added to `main.py` that control the debiasing objective
6. Minor quality of life improvements and tqdm progress monitoring
