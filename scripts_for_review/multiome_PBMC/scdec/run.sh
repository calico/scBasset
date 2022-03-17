
# run multiome_PBMC
python main_clustering.py --data multiome_PBMC --train True --no_label
python eval.py --data multiome_PBMC --timestamp 20220217_073452 --epoch 30000 --no_label
