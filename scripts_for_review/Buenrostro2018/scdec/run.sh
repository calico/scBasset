# run Buenrostro2018
python main_clustering.py --data Buenrostro2018 --train True --no_label
python eval.py --data Buenrostro2018 --timestamp 20220209_053810 --epoch 30000 --no_label

# Buenrostro2018: reduce K for batch correction
python main_clustering.py --data Buenrostro2018 --K 9 --train True --no_label
python main_clustering.py --data Buenrostro2018 --K 7 --train True --no_label
python main_clustering.py --data Buenrostro2018 --K 5 --train True --no_label
python main_clustering.py --data Buenrostro2018 --K 3 --train True --no_label
python eval.py --data Buenrostro2018 --timestamp 20220211_073107 --epoch 30000 --no_label
python eval.py --data Buenrostro2018 --timestamp 20220211_075905 --epoch 30000 --no_label
python eval.py --data Buenrostro2018 --timestamp 20220211_082604 --epoch 30000 --no_label
python eval.py --data Buenrostro2018 --timestamp 20220211_085258 --epoch 30000 --no_label
