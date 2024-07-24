# Fre-Painter: A Pytorch Implementation
## Pre-requisites
### 1. Clone our repository
```
git clone https://github.com/FrePainter/code.git
cd code
```
### 2. Install python requirements
```
pip install -r requirements.txt
``` 
## Preprocessing
### 1. Download dataset
- [VCTK](https://datashare.ed.ac.uk/handle/10283/2651)  
- [LibriTTS](https://www.openslr.org/60/)
### 2. Preprocessing for pre-training
```
INPUT_DIR=/data1/DB/LibriTTS/LibriTTS
OUTPUT_DIR=/data1/projects/frepainter/preprocessed_data/LibriTTS
CUDA_VISIBLE_DEVICES=0,1 python preprocess.py -i $INPUT_DIR -o $OUTPUT_DIR
```
### 3. Preprocessing for fine-tuning
```
INPUT_DIR=/data1/DB/VCTK/VCTK/data/VCTK-Corpus
OUTPUT_DIR=/data1/projects/frepainter/preprocessed_data/VCTK
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python preprocess.py -i $INPUT_DIR -o $OUTPUT_DIR --save_audio
```
## Pre-training
```
PT_MODEL_NAME=pretrain_80
MASK_RATIO=0.8
CUDA_VISIBLE_DEVICES=0,1 python pretrain.py -m $PT_MODEL_NAME -r $MASK_RATIO
```
## Fine-tuning
```
FT_MODEL_NAME=finetune_random
PT_MODEL_NAME=pretrain_80
CUDA_VISIBLE_DEVICES=0,1 python finetune.py -m $FT_MODEL_NAME -p $PT_MODEL_NAME
```
## Inference of testset
### 1. Generation of testset
```
INPUT_DIR=[Directory of VCTK]
TESTSET_DIR=./dataset/testset
CUDA_VISIBLE_DEVICES=0,1 python generate_testset.py -m $INPUT_DIR -o $OUTPUT_DIR
```
### 2. Inference of audio
```
FT_MODEL_NAME=finetune_random
TESTSET_DIR=./dataset/testset
CUDA_VISIBLE_DEVICES=0,1 python inference_for_test.py -m $FT_MODEL_NAME -d $TESTSET_DIR
```
## Inference with the pre-trained model

```
sh download_checkpoint.sh
MODEL_NAME=pt_rd_80_ft_ub_mrv2
DATA_DIR=[Directory or audio file]
OUTPUT_DIR=[output directory]
EXT=wav
CUDA_VISIBLE_DEVICES=0,1 python inference_from_audio.py -m $MODEL_NAME -d $DATA_DIR -o $OUTPUT_DIR -e $EXT
```
## Referece
- https://github.com/rishikksh20/AudioMAE-pytorch
- https://github.com/jaywalnut310/vits
- https://github.com/mindslab-ai/nuwave2
- https://github.com/haoheliu/versatile_audio_super_resolution
