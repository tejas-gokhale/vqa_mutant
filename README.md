# VQA-MUTANT 

Code for **MUTANT: A Training Paradigm for Out-of-Distribution Generalization in Visual Question Answering**
- _Tejas Gokhale, Pratyay Banerjee, Chitta Baral, Yezhou Yang_
- EMNLP 2020 , https://arxiv.org/abs/2009.08566

In this paper, we use two backbone models: 
1. LXMERT (Tan et al. EMNLP 2019), and 
2. UpDn (Anderson et al. CVPR 2017).
The instructions for running each can be found in respective folders. 

## LXMERT
`cd lxmert`

### Download data and pre-trained model
(features and json files for mutant samples) 
- download image features and .json files from: `https://drive.google.com/drive/folders/1QldD1BPDK8cojlrq2OhMeUYlmNo8RFGl?usp=sharing`
- put image features under `data/mutant_imgfeat/`
- put the .json files under `data/vqa/`
- download pre-trained model from: 

### Command to test pre-trained model
### Command to run training:
```
bash run/vqa_finetune_mutant.bash 0 debug
```
where 0 is the gpu-id and debug is the name of the experiment (these are command line arguments that can be changed according to your preference.)

## Mutant Masks and Images:
These can be found as 3 directories in the Google drive link: https://drive.google.com/drive/folders/1QldD1BPDK8cojlrq2OhMeUYlmNo8RFGl?usp=sharing

- the masks dir contains the binary masks that we use to create each mutated image
- the 'masked' dir contains images with the masked region cropped out
- the 'images' dir contains the final mutated image

Each of these directories contains 'train2014' and 'val2014' according to the standard VQA split.
The example below shows the filename convention we used
![example image](COCO_train2014_000000000025_number_giraffe_2.jpg)

*filename*    COCO_train2014_000000000025_number_giraffe_2
*convention*  original COCO filename_type-of-question_name-of-object-removed_number of object instances removed
