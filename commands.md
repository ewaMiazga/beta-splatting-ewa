# To train for beta mode
bash```
python train.py --source_path ../data-ready/tandt/truck/ --model_path output_truck_sb_2 --rendering_mode beta --eval
```

# To eval for beta mode
bash```
python eval.py --model_path output_truck_sb_2 --iteration 30000 --rendering_mode beta --sh_degree 2 --source_path ../data-ready/tandt/truck
```
# To train for nasg mode
bash``` 
python train.py --source_path ../data-ready/tandt/truck/ --model_path output_truck_nasg_1 --rendering_mode gmm --sb_number 1 --eval
```

# to eval for gmm mode
bash```
python eval.py --model_path output_truck_nasg_1 --iteration 30000 --rendering_mode gmm --sb_number 1 --source_path ../data-ready/tandt/truck
```