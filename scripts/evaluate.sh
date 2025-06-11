
# echo "~~~~~~~~~~~Running Endonerf Cutting experiments:~~~~~~~~~~~ " >> results.out

# echo "~~~~~~~~~~~Surgedepth:~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Endonerf/cutting_tissues_twice-20241203T074853Z-001/cutting_tissues_twice/" --load_weights_folder './models/SurgeDepth' --model_type "surgedepth" --width 336 --height 336 --eval_mono --eval_split 'endonerf' --endovis_split test --visualize_depth --vis_folder 'test_depth_vis/En_cut' >> results.out
# echo "~~~~~~~~~~~EndoDAC~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Endonerf/cutting_tissues_twice-20241203T074853Z-001/cutting_tissues_twice/" --load_weights_folder './models/EndoDAC_fullmodel' --model_type "endodac" --eval_mono --eval_split 'endonerf' --endovis_split test --visualize_depth --vis_folder 'test_depth_vis/En_cut'>> results.out
# echo "~~~~~~~~~~~DAv2~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Endonerf/cutting_tissues_twice-20241203T074853Z-001/cutting_tissues_twice/" --load_weights_folder './models/depth_anything_v2' --model_type "depthanything_v2" --width 518 --height 518 --eval_mono --eval_split 'endonerf' --endovis_split test --visualize_depth --vis_folder 'test_depth_vis/En_cut'>> results.out
# echo "~~~~~~~~~~~DAv1~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Endonerf/cutting_tissues_twice-20241203T074853Z-001/cutting_tissues_twice/" --load_weights_folder './models/depth_anything_v1' --model_type "depthanything_v1" --width 518 --height 518 --eval_mono --eval_split 'endonerf' --endovis_split test --visualize_depth --vis_folder 'test_depth_vis/En_cut'>> results.out
# echo "~~~~~~~~~~~DinoV2~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Endonerf/cutting_tissues_twice-20241203T074853Z-001/cutting_tissues_twice/" --load_weights_folder './models/depth_anything_v2' --model_type "dino_v2" --width 518 --height 518 --eval_mono --eval_split 'endonerf' --endovis_split test --visualize_depth --vis_folder 'test_depth_vis/En_cut'>> results.out


# echo "~~~~~~~~~~~Running Endonerf Pulling experiments:~~~~~~~~~~~ (rerun for Dino)" >> results.out

# echo "~~~~~~~~~~~Surgedepth:~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Endonerf/pulling_soft_tissues-20241203T074946Z-001/pulling_soft_tissues/" --load_weights_folder './models/SurgeDepth' --model_type "surgedepth" --width 336 --height 336 --eval_mono --eval_split 'endonerf' --endovis_split test  --visualize_depth --vis_folder 'test_depth_vis/En_pull' >> results.out
# echo "~~~~~~~~~~~EndoDAC~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Endonerf/pulling_soft_tissues-20241203T074946Z-001/pulling_soft_tissues/" --load_weights_folder './models/EndoDAC_fullmodel' --model_type "endodac" --eval_mono --eval_split 'endonerf' --endovis_split test --visualize_depth --vis_folder 'test_depth_vis/En_pull' >> results.out
# echo "~~~~~~~~~~~DAv2~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Endonerf/pulling_soft_tissues-20241203T074946Z-001/pulling_soft_tissues/" --load_weights_folder './models/depth_anything_v2' --model_type "depthanything_v2" --width 518 --height 518 --eval_mono --eval_split 'endonerf' --endovis_split test --visualize_depth --vis_folder 'test_depth_vis/En_pull' >> results.out
# echo "~~~~~~~~~~~DAv1~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Endonerf/pulling_soft_tissues-20241203T074946Z-001/pulling_soft_tissues/" --load_weights_folder './models/depth_anything_v1' --model_type "depthanything_v1" --width 518 --height 518 --eval_mono --eval_split 'endonerf' --endovis_split test --visualize_depth --vis_folder 'test_depth_vis/En_pull' >> results.out
# echo "~~~~~~~~~~~DinoV2~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Endonerf/pulling_soft_tissues-20241203T074946Z-001/pulling_soft_tissues/" --load_weights_folder './models/depth_anything_v2' --model_type "dino_v2" --width 518 --height 518 --eval_mono --eval_split 'endonerf' --endovis_split test --visualize_depth --vis_folder 'test_depth_vis/En_pull'>> results.out


# echo "~~~~~~~~~~~SCARED experiments:~~~~~~~~~~~ (test) " >> results.out
# echo "~~~~~~~~~~~Surgedepth:~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/SCARED" --load_weights_folder './models/SurgeDepth' --model_type "surgedepth" --width 336 --height 336 --eval_mono --eval_split 'endovis' --endovis_split test  --vis_folder 'test_depth_vis/scared'>> results.out
# echo "~~~~~~~~~~~EndoDAC~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/SCARED" --load_weights_folder './models/EndoDAC_fullmodel' --model_type "endodac" --eval_mono --eval_split 'endovis' --endovis_split test   --vis_folder 'test_depth_vis/scared'>> results.out
# echo "~~~~~~~~~~~DAv2~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/SCARED" --load_weights_folder './models/depth_anything_v2' --model_type "depthanything_v2" --width 518 --height 518 --eval_mono --eval_split 'endovis' --endovis_split test  --vis_folder 'test_depth_vis/scared'>> results.out
# echo "~~~~~~~~~~~DAv1~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/SCARED" --load_weights_folder './models/depth_anything_v1' --model_type "depthanything_v1" --width 518 --height 518 --eval_mono --eval_split 'endovis' --endovis_split test --vis_folder 'test_depth_vis/scared' >> results.out
# echo "~~~~~~~~~~~DinoV2~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/SCARED" --load_weights_folder './models/depth_anything_v1' --model_type "dino_v2" --width 518 --height 518 --eval_mono --eval_split 'endovis' --endovis_split test  --vis_folder 'test_depth_vis/scared'>> results.out



# echo "~~~~~~~~~~~SCARED experiments:~~~~~~~~~~~ (train)  " >> results.out
# echo "~~~~~~~~~~~Surgedepth:~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/SCARED" --load_weights_folder './models/SurgeDepth' --model_type "surgedepth" --width 336 --height 336 --eval_mono --eval_split 'endovis' --endovis_split train >> results.out
# echo "~~~~~~~~~~~EndoDAC~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/SCARED" --load_weights_folder './models/EndoDAC_fullmodel' --model_type "endodac" --eval_mono --eval_split 'endovis' --endovis_split train >> results.out
# echo "~~~~~~~~~~~DAv2~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/SCARED" --load_weights_folder './models/depth_anything_v2' --model_type "depthanything_v2" --width 518 --height 518 --eval_mono --eval_split 'endovis' --endovis_split train >> results.out
# echo "~~~~~~~~~~~DAv1~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/SCARED" --load_weights_folder './models/depth_anything_v1' --model_type "depthanything_v1" --width 518 --height 518 --eval_mono --eval_split 'endovis' --endovis_split train >> results.out
# echo "~~~~~~~~~~~DinoV2~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/SCARED" --load_weights_folder './models/depth_anything_v1' --model_type "dino_v2" --width 518 --height 518 --eval_mono --eval_split 'endovis' --endovis_split train >> results.out





# echo "~~~~~~~~~~~SCARED experiments:~~~~~~~~~~~ (val) " >> results.out
# echo "~~~~~~~~~~~Surgedepth:~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/SCARED" --load_weights_folder './models/SurgeDepth' --model_type "surgedepth" --width 336 --height 336 --eval_mono --eval_split 'endovis' --endovis_split val  >> results.out
# echo "~~~~~~~~~~~EndoDAC~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/SCARED" --load_weights_folder './models/EndoDAC_fullmodel' --model_type "endodac" --eval_mono --eval_split 'endovis' --endovis_split val >> results.out
# echo "~~~~~~~~~~~DAv2~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/SCARED" --load_weights_folder './models/depth_anything_v2' --model_type "depthanything_v2" --width 518 --height 518 --eval_mono --eval_split 'endovis' --endovis_split val >> results.out
# echo "~~~~~~~~~~~DAv1~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/SCARED" --load_weights_folder './models/depth_anything_v1' --model_type "depthanything_v1" --width 518 --height 518 --eval_mono --eval_split 'endovis' --endovis_split val >> results.out
# echo "~~~~~~~~~~~DinoV2~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/SCARED" --load_weights_folder './models/depth_anything_v1' --model_type "dino_v2" --width 518 --height 518 --eval_mono --eval_split 'endovis' --endovis_split val >> results.out




# echo "~~~~~~~~~~~Hamlyn experiments:~~~~~~~~~~~ " >> results.out

# echo "~~~~~~~~~~~Surgedepth:~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Hamlyn" --load_weights_folder './models/SurgeDepth' --model_type "surgedepth" --width 336 --height 336 --eval_mono --eval_split 'hamlyn'  --vis_folder 'test_depth_vis/hamlyn'  >> results.out
# echo "~~~~~~~~~~~EndoDAC~~~~~~~~~~~">> results.out
python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Hamlyn" --load_weights_folder './models/EndoDAC_fullmodel' --model_type "endodac" --eval_mono --eval_split 'hamlyn'  --visualize_depth --vis_folder 'test_depth_vis/hamlyn/tst'>> results.out
# echo "~~~~~~~~~~~DAv2~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Hamlyn" --load_weights_folder './models/depth_anything_v2' --model_type "depthanything_v2" --width 518 --height 518 --eval_mono --eval_split 'hamlyn'  --vis_folder 'test_depth_vis/hamlyn' >> results.out
# echo "~~~~~~~~~~~DAv1~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Hamlyn" --load_weights_folder './models/depth_anything_v1' --model_type "depthanything_v1" --width 518 --height 518 --eval_mono --eval_split 'hamlyn'  >> results.out
# echo "~~~~~~~~~~~DinoV2~~~~~~~~~~~">> results.out
# python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Hamlyn" --load_weights_folder './models/depth_anything_v2' --model_type "dino_v2" --width 518 --height 518 --eval_mono --eval_split 'hamlyn' --vis_folder 'test_depth_vis/hamlyn'  >> results.out




# # echo "~~~~~~~~~~~Hamlyn experiments:~~~~~~~~~~~ " #>> results.out

# # echo "~~~~~~~~~~~Surgedepth:~~~~~~~~~~~" #>> results.out
# # python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Hamlyn" --load_weights_folder './models/SurgeDepth' --model_type "surgedepth" --width 336 --height 336 --eval_mono --eval_split 'hamlyn' --visualize_depth --vis_folder 'test_depth_vis' # >> results.out
# # # echo "~~~~~~~~~~~EndoDAC~~~~~~~~~~~">> results.out
# # # python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Hamlyn" --load_weights_folder './models/EndoDAC_fullmodel' --model_type "endodac" --eval_mono --eval_split 'hamlyn'  >> results.out
# # echo "~~~~~~~~~~~DAv2~~~~~~~~~~~">> results.out
# # python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Hamlyn" --load_weights_folder './models/depth_anything_v2' --model_type "depthanything_v2" --width 518 --height 518 --eval_mono --eval_split 'hamlyn'  --visualize_depth --vis_folder 'test_depth_vis' #  >> results.out
# # echo "~~~~~~~~~~~DAv1~~~~~~~~~~~">> results.out
# # python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Hamlyn" --load_weights_folder './models/depth_anything_v1' --model_type "depthanything_v1" --width 518 --height 518 --eval_mono --eval_split 'hamlyn'  >> results.out
# # echo "~~~~~~~~~~~DinoV2~~~~~~~~~~~">> results.out
# # python3 evaluate_depth.py --data_path "/media/thesis_ssd/data/Hamlyn" --load_weights_folder './models/depth_anything_v2' --model_type "dino_v2" --width 518 --height 518 --eval_mono --eval_split 'hamlyn'  >> results.out
