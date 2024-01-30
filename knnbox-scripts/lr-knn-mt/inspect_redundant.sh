export OMP_WAIT_POLICY=PASSIVE

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/medical
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/it

python $PROJECT_PATH/knnbox-scripts/common/generate.py $PROJECT_PATH/data-bin/it \
--task translation \
--path $PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt \
--dataset-impl mmap \
--beam 4 \
--lenpen 0.6 \
--max-len-a 1.2 \
--max-len-b 10 \
--source-lang de \
--target-lang en  \
--gen-subset test \
--max-tokens 2048  \
--scoring sacrebleu  \
--tokenizer moses \
--remove-bpe \
--user-dir $PROJECT_PATH/knnbox/models \
--arch vanilla_knn_mt_inspect_redundant@transformer_wmt19_de_en \
--knn-mode inference  \
--knn-datastore-path $PROJECT_PATH/datastore/vanilla/it \
--knn-k 8 \
--knn-lambda 0.7 \
--knn-temperature 10.0