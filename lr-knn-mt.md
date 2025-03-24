
## Results in table 4

For our results, you need to build datastores first.

Required by vanilla,lr:

```
cd knnbox-scripts/vanilla-knn-mt
bash build_datastore.sh  
```

Train selector:

```
cd knnbox-scripts/lr-knn-mt
python train_less_retrieve.py --dataset it
python train_less_retrieve.py --dataset koran
python train_less_retrieve.py --dataset law
python train_less_retrieve.py --dataset medical
```

You can specific gpu id by passing additional argument

```
--single-gpu-index 1
```

to use anther GPU.

and make inference:

```
cd knnbox-scripts/lr-knn-mt
python inference.py --dataset it --model vanilla
python inference.py --dataset it --model lr
``

You may need to **edit** this build_datastore.sh and change another dataset, generated datastore will be placed in datastore folder.

Required adaptive, lr_adaptive:

```
cd knnbox-scripts/adaptive-knn-mt
bash build_datastore.sh
bash train_metak.sh
```

```
cd knnbox-scripts/lr-knn-mt
python inference.py --dataset it --model adaptive
python inference.py --dataset it --model lr_adaptive
```

Also, edition may be required.


Required by pck, lr_pck:

```
cd knnbox-scripts/pck-knn-mt
# step 1. build datastore 
bash build_datastore.sh
# step 2. train reduction network
bash train_reduct_network.sh
# step 3. reduct datastore's key dimension using trained network
bash reduct_datastore_dim.sh
# step 4. train meta-k network
bash train_metak.sh
```

```
cd knnbox-scripts/lr-knn-mt
python inference.py --dataset it --model pck
python inference.py --dataset it --model lr_pck
```

Thus, you could get results in table 4.

## Results in table 3

Edit knnbox/models/models/less_retrieve_knn_mt.py and vanilla_knn_mt.py

Find all 

```
#record_timer_start(self.retrieve_timer)
```

and 

```
#record_timer_start(self.retrieve_timer)
```

in the code and uncomment them.

In vanilla_knn_mt.py, they are line 138,140,161,164

In less_retrieve_knn_mt.py, they are line 459,471,530,551,554,566

and run

```
python inference.py --dataset it --model vanilla --test-knn-overhead
python inference.py --dataset it --model lr --test-knn-overhead
python inference.py --dataset koran --model vanilla --test-knn-overhead
python inference.py --dataset koran --model lr --test-knn-overhead
python inference.py --dataset law --model vanilla --test-knn-overhead
python inference.py --dataset law --model lr --test-knn-overhead
python inference.py --dataset medical --model vanilla --test-knn-overhead
python inference.py --dataset medical --model lr --test-knn-overhead
```

then you will get the results in table 3.

After testing knn overhead time, you should comment those code, they will introduce slight latency for inference due to torch.cuda.synchronize()

## Results in table 5

```
cd knnbox-scripts/lr-knn-mt
python test_metrics.py --dataset it
```

also run it on other datasets, you could get results in table 5.

## Results in table 1 and figure 2

```
cd knnbox-scripts/lr-knn-mt
bash inspect_redundant.sh
```

Also, you may need to edit this inspect_redundant.sh to switch datasets.

## Results in table 6, 7 (Ablation for translation loss)

Train selectors and infernece with --no-translation-loss

```
python train_less_retrieve.py --dataset it --no-translation-loss
python inference.py --dataset it --model lr --no-translation-loss
python train_less_retrieve.py --dataset koran --no-translation-loss
python inference.py --dataset koran --model lr --no-translation-loss
python train_less_retrieve.py --dataset law --no-translation-loss
python inference.py --dataset law --model lr --no-translation-loss
python train_less_retrieve.py --dataset medical --no-translation-loss
python inference.py --dataset medical --model lr --no-translation-loss
```

