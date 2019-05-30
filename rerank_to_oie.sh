#!/bin/bash

tag_model=$1
rerank_model=$2

# create dir to hold oie model
mkdir -p ${rerank_model}/tag_model

# copy original oie model
cp ${tag_model}/model.tar.gz ${rerank_model}/tag_model/.

# uncompress
pushd ${rerank_model}/tag_model/
tar xzf model.tar.gz

# replace the weight
rm weights.th
popd
cp ${rerank_model}/best.th ${rerank_model}/tag_model/weights.th
