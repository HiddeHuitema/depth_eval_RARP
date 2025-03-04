

python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/SCARED" --load_weights_folder './models/depth_pro' --model_type "depthpro" --width 518 --height 518 --eval_mono --eval_split 'endovis' --endovis_split train

python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Hamlyn" --load_weights_folder './models/depth_pro' --model_type "depthpro" --width 518 --height 518 --eval_mono --eval_split 'hamlyn'