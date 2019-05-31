#!/bin/bash
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --time=0

set -e

model_env_name=oie_rank
eval_env_name=sup_oie

niter=$1                # number of iterations
data_dir=$2/data        # data dir.
model_dir=$2/model      # model dir. Initial model should be put in
                        # ${model_dir}/iter0/tag_model/model.tar.gz
                        # (both compressed and uncompressed)
eval_dir=$2/eval        # evaluation dir. Initial extractions should be put in
                        # ${eval_dir}/iter0/tag_model
rerank_conf=$3          # training config file
beam=$4                 # size of beam search

# make path absolute
pwd_dir=$(pwd)
data_dir=${pwd_dir}/${data_dir}
model_dir=${pwd_dir}/${model_dir}
eval_dir=${pwd_dir}/${eval_dir}

# set up conda
conda_act=${CONDA_HOME}/bin/activate
conda_dea=${CONDA_HOME}/bin/deactivate

source $conda_act $model_env_name

for (( e=1; e<=$niter; e++ ))
do
    echo "======================== Iter $e ========================"

    cur_dir=iter$((e-1))
    next_dir=iter${e}

    echo "beam search to generate extractions"

    mkdir -p ${data_dir}/${next_dir}
    for split in train dev
    do
        python openie_extract.py \
            --model ${model_dir}/${cur_dir}/tag_model \
            --inp data/sent/oie2016_${split}.txt \
            --out ${data_dir}/${next_dir}/oie2016.${split}.beam \
            --beam_search ${beam} \
            --keep_one
    done

    echo "annotate extractions, convert them to sup-oie conll format, and combine them with the previous epoch"

    pushd ../supervised-oie/supervised-oie-benchmark
    source $conda_dea
    source $conda_act $eval_env_name
    for split in train dev
    do
        last_conll=${data_dir}/${cur_dir}/oie2016.${split}.iter.conll
        this_beam=${data_dir}/${next_dir}/oie2016.${split}.beam
        this_beam_conll=${data_dir}/${next_dir}/oie2016.${split}.beam.conll
        this_conll=${data_dir}/${next_dir}/oie2016.${split}.iter.conll

        python benchmark.py \
            --gold=oie_corpus/${split}.oie.orig.correct.head \
            --out=/dev/null \
            --tabbed=${this_beam} \
            --label=${this_beam_conll} \
            --predArgHeadMatch

        if [ -f "$last_conll" ]
        then
            python combine_conll.py -inp=${last_conll}:${this_beam_conll} -out=${this_conll}
        else
            # combine with self to avoid dup extractions
            python combine_conll.py -inp=${this_beam_conll}:${this_beam_conll} -out=${this_conll}
        fi
    done
    source $conda_dea
    source $conda_act $model_env_name
    popd

    echo "prepare rank-aware training config file"

    conf_data_dir=${data_dir}/${next_dir} # training data dir
    conf_model_dir=${model_dir}/${cur_dir}/tag_model # initial model dir
    conf_conf=${conf_data_dir}/iter_conf.jsonnet # generated training conf
    sed "s|ITER_DATA_ROOT|${conf_data_dir}|g" ${rerank_conf} > ${conf_conf}
    sed -i "s|ITER_MODEL_ROOT|${conf_model_dir}|g" ${conf_conf}
    mkdir -p ${model_dir}/${next_dir}

    echo "iterative rank-aware training"

    allennlp train ${conf_conf} \
        --include-package rerank \
        --serialization-dir ${model_dir}/${next_dir}

    echo "convert extractions to sup-oie conll format"

    eval_from=${eval_dir}/${cur_dir}/tag_model
    eval_to=${eval_dir}/${next_dir}
    mkdir -p ${eval_to}
    pushd ../supervised-oie/supervised-oie-benchmark
    source $conda_dea
    source $conda_act $eval_env_name

    python benchmark.py \
        --gold=oie_corpus/test.oie.orig.correct.head \
        --out=/dev/null \
        --tabbed=${eval_from}/oie2016.txt \
        --label=${eval_to}/oie2016.txt.conll \
        --predArgHeadMatch

    source $conda_dea
    source $conda_act $model_env_name
    popd

    echo "rerank extractions generated from previous model"

    python rerank.py \
        --model ${model_dir}/${next_dir}/model.tar.gz \
        --inp ${eval_to}/oie2016.txt.conll:${eval_from}/oie2016.txt \
        --out ${eval_to}/oie2016.txt

    echo "convert rerank model to openie model"

    ./rerank_to_oie.sh ${model_dir}/iter0/tag_model ${model_dir}/${next_dir}

    echo "generate extractions using the resulting model"

    mkdir -p ${eval_to}/tag_model
    python openie_extract.py \
        --model ${model_dir}/${next_dir}/tag_model \
        --inp data/sent/oie2016_test.txt \
        --out ${eval_to}/tag_model/oie2016.txt \
        --keep_one
done
