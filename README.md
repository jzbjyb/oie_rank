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
    pip install allennlp==0.8.1
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
    cd oie_rank
    mkdir pretrain/elmo/
    cd pretrain/elmo/
    wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
    wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
    ```

2. Download pre-trained RnnOIE and its extractions. `gdown.pl` can be download from [here](https://github.com/circulosmeos/gdown.pl).
    ```bash
    cd oie_rank
    mkdir pretrain/rnnoie/
    cd pretrain/rnnoie/
    gdown.pl https://drive.google.com/open?id=1ykuxnIE1vkhJeq-81UPtAs6K6_h3Vcw3 model.tar.gz  # download model
    gdown.pl https://drive.google.com/open?id=1Mn2ptAAiB5rTRg3GYMtBk9lrFcC3Qb4R oie2016.txt  # download extractions
    ```

## Iterative rank-aware learning

First, create data, model, and evaluation directories for iterative training.
```bash
./init_iter.sh
```

Then, run iterative rank-aware training (3 iterations with beam size of 5).
```bash
CONDA_HOME=PATH_TO_YOUR_CONDA_HOME ./iter.sh \
    3 iter/data iter/model iter/eval training_config/iter.jsonnet 5
```

The extractions generated at `i`-th iteration are saved to `iter/eval/iteri/tag_model/`.
The reranking results of extractions generated at `i-1`-th iteration are saved to `iter/eval/iteri/`.

## Train and evaluate RnnOIE

In addition to using pretrained RnnOIE model, you can also use the following commands to train from scratch and evaluate it.

```bash
cd oie_rank
conda activate oie_rank
# train
allennlp train training_config/oie.jsonnet \
    --include-package rerank \
    --serialization-dir MODEL_DIR
# generate extractions
python openie_extract.py \
    --model MODEL_DIR/model.tar.gz \
    --inp data/sent/oie2016_test.txt \
    --out RESULT_DIR/oie2016.txt \
    --keep_one
# evaluate
cd ../supervised-oie/supervised-oie-benchmark/
conda activate sup_oie
python benchmark.py \
    --gold=oie_corpus/test.oie.orig.correct.head \
    --out=/dev/null \
    --tabbed=RESULT_DIR/oie2016.txt \
    --predArgHeadMatch
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
