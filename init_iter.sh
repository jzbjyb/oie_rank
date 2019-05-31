#!/bin/bash

root_dir=$1

mkdir ${root_dir}
mkdir ${root_dir}/data  # data root dir
mkdir ${root_dir}/model  # model root dir
mkdir ${root_dir}/eval  # evaluation root dir

# copy and uncompress initial RnnOIE model
mkdir -p ${root_dir}/model/iter0/tag_model
cp pretrain/rnnoie/model.tar.gz ${root_dir}/model/iter0/tag_model
pushd ${root_dir}/model/iter0/tag_model
tar zxvf model.tar.gz
popd

# copy initial extractions
mkdir -p ${root_dir}/eval/iter0/tag_model
cp pretrain/rnnoie/oie2016.txt ${root_dir}/eval/iter0/tag_model
