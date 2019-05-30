# Iterative Rank-Aware OIE

This repository contains data and code to reproduce the experiments in:

Improving Open Information Extraction via Iterative Rank-Aware Learning.

by Zhengbao Jiang, Pengcheng Yin, and Graham Neubig.

## Install

Our model implementation is based on [AllenNLP](https://github.com/allenai/allennlp), and evaluation is based on [supervised-oie](https://github.com/gabrielStanovsky/supervised-oie).

1. Prepare the model repository:
    ```bash
    git clone https://github.com/jzbjyb/oie_rank.git  # clone the model repository
    cd oie_rank
    conda create -n oie_rank python=3.6
    conda activate oie_rank
    conda install allennlp=0.8.2
    conda deactivate
    cd ..
    ```

2. Prepare the evaluation repository:
    ```bash
    git clone https://github.com/jzbjyb/supervised-oie.git  # clone the evaluation repository
    cd supervised-oie
    git reset --hard ef66de761a5230a7905918b4c7ce87147369bf33  # rollback to the evaluation metrics used in this paper
    conda create -n sup_oie python=2.7
    conda activate sup_oie
    pip install -r requirements_python2.txt
    conda deactivate
    cd ..
    ```

## Download pre-trained models

1. Download pre-trained ELMo (weights and options).
    ```bash
    mkdir pretrain/elmo/
    cd pretrain/elmo/
    wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
    wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
    ```

2. Download pre-trained RnnOIE and its extractions. `gdown.pl` can be download from [here](https://github.com/circulosmeos/gdown.pl).
    ```bash
    mkdir pretrain/rnnoie/
    cd pretrain/rnnoie/
    gdown.pl https://drive.google.com/open?id=1ykuxnIE1vkhJeq-81UPtAs6K6_h3Vcw3 model.tar.gz  # download model
    gdown.pl https://drive.google.com/open?id=1Mn2ptAAiB5rTRg3GYMtBk9lrFcC3Qb4R oie2016.txt  # download extractions
    ```

## Train and evaluate RnnOIE

```bash
conda activate oie_rank
allennlp train training_config/oie.jsonnet \
    --include-package rerank \
    --serialization-dir MODEL_DIR  # train
python scripts/openie_extract.py \
    --model MODEL_DIR/model.tar.gz \
    --inp data/sent/oie2016_test.txt \
    --out RESULT_DIR/oie2016.txt \
    --keep_one  # generate extractions
cd ../supervised-oie/supervised-oie-benchmark/
conda activate sup_oie
python benchmark.py \
    --gold=oie_corpus/test.oie.orig.correct.head \
    --out=/dev/null \
    --tabbed=RESULT_DIR/oie2016.txt \
    --predArgHeadMatch  # evaluate
```

## Iterative rank-aware learning

First, create data, model, and evaluation directories for iterative training.
```bash
mkdir iter
mkdir iter/data  # data root dir
mkdir iter/model  # model root dir
mkdir -p iter/model/iter0/tag_model
cp pretrain/rnnoie/model.tar.gz iter/model/iter0/tag_model
pushd iter/model/iter0/tag_model
tar zxvf model.tar.gz
popd  # copy and uncompress inital oie model
mkdir iter/eval  # evaluation root dir
mkdir -p iter/eval/iter0/tag_model
cp pretrain/rnnoie/oie2016.txt iter/eval/iter0/tag_model  # copy initial extractions
```

Then, run iterative rank-aware training (3 iterations with beam size of 5).
```bash
./iter.sh 3 iter/data iter/model iter/eval training_config/iter.jsonnet 5
```

## Reference

```
@inproceedings{jiang19acl,
    title = {Improving Open Information Extraction via Iterative Rank-Aware Learning},
    author = {Zhengbao Jiang and Pengcheng Yin and Graham Neubig},
    booktitle = {The 57th Annual Meeting of the Association for Computational Linguistics (ACL)},
    address = {Florence, Italy},
    month = {July},
    year = {2019}
}
```