# NOTICE:

This repository contains codes used for the experiments in Master's thesis of BJTU 22120416. 

Many thanks to the upstream repository [NJUNLP/knn-box](https://github.com/NJUNLP/knn-box), which helped me successfully reproduce the baseline models and implement the new models based on it.

Relevant citations of manuscripts(GBT-7714 format):

```
SHI X Y, LIANG Y L, XU J A, et al. Towards Faster k-Nearest-Neighbor Machine Translation[J]. Advances in Artificial Intelligence and Machine Learning, 2024.

齐睿,石响宇,满志博,等.融合确定性因子及区域密度的k-最近邻机器翻译方法[C]. 中国计算语言学大会(CCL), 2024.
```

# Less-Retrive KNN-MT

This is a fork of KNN-BOX, so also see [KNNBOX.md](KNNBOX.md). 

Major changes:

1. Add less retreive knn-mt method [code](knnbox/models/less_retrieve_knn_mt.py)
2. Add MoE knn-mt method [code](knnbox/models/moe_knn_mt.py)
3. Change codes of fairseq 0.10.1 to be compatible with python3.9 (Because python 3.7 has been deprecated now in year 2025, VSCode can't work with python 3.7 in latest version.)
4. Add after_train_hook(called from fairseq train.py) and after_inference_hook(called from fairseq generate.py)

# Prepare

```bash
conda create -n knn-box python=3.9
conda activate knn-box
conda install pytorch==1.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip uninstall numpy
pip install numpy==1.21.6 scikit-learn==1.0.2 editdistance sacremoses elasticsearch
conda install faiss-gpu -c pytorch
pip install --editable ./
```

If some other packages is missing, just pip install it. If the conda pytorch does not work, you can use pip: 
```bash
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

These code also supprt the research of CF-KNN-MT(the 2-rd relevant manuscript)

The model implementations are in knnbox/models/models/less_retrieve_knn_mt.py.

Most of the results have been recorded in knnbox-scripts/lr-knn-mt/infernece_results folder.

Some results requre re-run the programs to see.

### Get dataset and base neural model:

```
cd knnbox-scripts
bash prepare_dataset_and_model.sh
```

The baseline model will be placed at pretrain-models/wmt19.de-en.

Datasets are in data-bin folder.

This dataset is a multi-domain corpus.

### Run baseline model (Pure neural machine translation model):

```
cd knnbox-scripts/lr-knn-mt
python inference.py --dataset it --model base
```

Here --dataset parameter could be one of: 
```
it,koran,law,medical
```
 --model could be one of 
```
base,vanilla,lr,adaptive,pck,lr_adaptive,lr_pck
```

wehre

```
base = Pure neural model
vanilla = Vanilla KNN-MT
lr = Vanilla + Selector
adaptive = Adaptive KNN-MT
lr_adaptive = Adaptive + Selector
pck = PCK KNN-MT
lr_pck = PCK + Selector
```

If need to measure time, add --run-3-time argument to inference.py, it will run an experiment three times and averge the time.

For lr-knn-mt, see details in [lr-knn-mt.md](lr-knn-mt.md)
