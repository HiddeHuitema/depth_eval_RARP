## Evaluating depth estimation on surgical footage

This git repo is for comparing the depth estimation performance of several different State of the Art models on surgical data. \
The code is largely based on the EndoDAC repo (https://github.com/BeileiCui/EndoDAC?tab=readme-ov-file), which in turn was adapted from AF-SFMLeaner (https://github.com/ShuweiShao/AF-SfMLearner)


Prepare env:
conda env create -f eval_env_2.yml
mim install mmcv-full


Run evaluation code on EndoDAC model:\
python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/SCARED" --load_weights_folder './models/EndoDAC_fullmodel' --eval_mono


Extract frames from mp4 (first create rgb folder):\
find . -name "*.mp4" -print0 | xargs -0 -I {} sh -c 'output_dir=$(dirname "$rgb"); ffmpeg -i "$1" "rgb/%10d.png"' _ {}


Extract gt depths: \
python3 export_gt_depth.py --data_path "/media/thesis_ssd/data/SCARED" --split "endovis" --useage eval
