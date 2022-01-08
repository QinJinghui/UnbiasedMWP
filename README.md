## bert2tree baseline:
```
python
```

## bert2tree + EMPM:
```
CUDA_VISIBLE_DEVICES=0 python run_bert2tree_em.py --save_path model/debug/ --n_epochs 50 --processed_path processed_data/unbias_ori.pkt --processed
```