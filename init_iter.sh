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
cp pretrain/rnnoie/oie2016.txt iter/eval/iter0/tag_model
