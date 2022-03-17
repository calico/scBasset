
# run multiome_mouse_brain
python main_clustering.py --data multiome_mousebrain --train True --no_label
python eval.py --data multiome_mousebrain --timestamp 20220217_081022 --epoch 30000 --no_label
