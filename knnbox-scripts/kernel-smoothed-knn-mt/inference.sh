:<<!
[script description]: use kernel smoothed knn-mt to translate 
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/medical
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/medical
COMBINER_LOAD_DIR=$PROJECT_PATH/save-models/combiner/kernel_smooth/medical


CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common/generate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
--gen-subset test \
--model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
--max-tokens 2048 \
--scoring sacrebleu \
--tokenizer moses --remove-bpe \
--arch kernel_smoothed_knn_mt@transformer_wmt19_de_en \
--user-dir $PROJECT_PATH/knnbox/models \
--knn-mode inference \
--knn-datastore-path $DATASTORE_LOAD_PATH \
--knn-k 16 \
--knn-combiner-path $COMBINER_LOAD_DIR \
