## Evaluating depth estimation on surgical footage

This git repo is for comparing the depth estimation performance of several different State of the Art models on surgical data. 
The code is largely based on the EndoDAC repo (https://github.com/BeileiCui/EndoDAC?tab=readme-ov-file), which in turn was adapted from AF-SFMLeaner (https://github.com/ShuweiShao/AF-SfMLearner)

Run evaluation code on EndoDAC model:
python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/SCARED" --load_weights_folder './models/EndoDAC_fullmodel' --eval_mono