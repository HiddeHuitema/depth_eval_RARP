from __future__ import absolute_import, division, print_function

import os
import cv2
import datasets.scared_dataset
import datasets.endonerf_data
import datasets.hamlyn_dataset
import numpy as np
from tqdm import tqdm
import time


import torch
from torch.utils.data import DataLoader
import torchvision.transforms as tf
from PIL import Image
import matplotlib.pyplot as plt
import scipy.stats as st

from utils.layers import disp_to_depth
from utils.utils import readlines, compute_errors,load_dinoV2_depth
from options import MonodepthOptions
import datasets
import models.encoders as encoders
import models.decoders as decoders
import models.endodac as endodac

from models.depth_anything_v2.dpt import DepthAnythingV2
from models.depth_anything_v1.dpt import DepthAnything
from models.SurgeDepth.dpt import SurgeDepth

from torchvision.transforms import Compose
from models.depth_anything_v1.util.transform import Resize, NormalizeImage, PrepareForNet

import torchvision


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)



def align_shift_and_scale(gt_disp, pred_disp):

    t_gt = np.median(gt_disp)
    # print(gt_disp)
    s_gt = np.mean(np.abs(gt_disp - t_gt))
    
    t_pred = np.median(pred_disp)
    s_pred = np.mean(np.abs(pred_disp - t_pred))
    # print(t_gt, s_gt, t_pred, s_pred)
    pred_disp_aligned = (pred_disp - t_pred) * (s_gt / s_pred) + t_gt

    return pred_disp_aligned, t_gt, s_gt, t_pred, s_pred

transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

def render_depth(disp):
    disp = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp = disp.astype(np.uint8)
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_INFERNO)
    return disp_color


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp
    
def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = array.min(),array.max()
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150
    counter = 0

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        if opt.model_type == 'endodac':
            depther_path = os.path.join(opt.load_weights_folder, "depth_model.pth")
            depther_dict = torch.load(depther_path)

        elif opt.model_type == 'depthanything_v2':
            depther_path = os.path.join(opt.load_weights_folder, "depth_anything_v2_vitb.pth")
            depther_dict = torch.load(depther_path, map_location='cpu')

        elif opt.model_type == 'depthanything_v1':
            depther_path = os.path.join(opt.load_weights_folder, "depth_anything_vitb14.pth")
            depther_dict = torch.load(depther_path, map_location='cpu')

        elif opt.model_type == 'surgedepth':
            depther_path = os.path.join(opt.load_weights_folder, "SurgeDepthStudent_V5.pth")
            depther_dict = torch.load(depther_path, map_location='cpu')

        elif opt.model_type == 'dino_v2':
            model = load_dinoV2_depth()

        elif opt.model_type == 'depthpro':
            from models.depth_pro.depth_pro import create_model_and_transforms
            model,transforms_depthpro = create_model_and_transforms(precision=torch.float16) # Takes up too much vram at float32          

            

        if opt.eval_split == 'endovis':
            filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
            dataset = datasets.scared_dataset.SCAREDRAWDataset(opt.data_path, filenames,
                                            opt.height, opt.width,
                                            [0], 4, is_train=False)
        elif opt.eval_split == 'endonerf':
            rgb_dir = os.path.join(opt.data_path,'images')
            depth_dir = os.path.join(opt.data_path,'depth')
            transforms = tf.Compose([tf.Resize((opt.height,opt.width),antialias = True),tf.ConvertImageDtype(torch.float32)])
            transforms_depth = tf.Compose([tf.ConvertImageDtype(torch.float32)])
            dataset = datasets.endonerf_data.EndoNerfDataset(rgb_dir,depth_dir,transforms,transforms_depth)
        elif opt.eval_split == 'hamlyn':
            rgb_dir = os.path.join(opt.data_path,'image01')
            depth_dir = os.path.join(opt.data_path,'depth01')
            transforms = tf.Compose([tf.Resize((opt.height,opt.width),antialias = True),tf.ConvertImageDtype(torch.float32)])
            transforms_depth = tf.Compose([tf.ToTensor()])
            dataset = datasets.hamlyn_dataset.HamlynDataset(rgb_dir,depth_dir,transforms,transforms_depth)



        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        if opt.model_type == 'endodac':
            depther = endodac.endodac(
                backbone_size = "base", r=opt.lora_rank, lora_type=opt.lora_type,
                image_shape=(opt.width,opt.height), pretrained_path=opt.pretrained_path,
                residual_block_indexes=opt.residual_block_indexes,
                include_cls_token=opt.include_cls_token)
            model_dict = depther.state_dict()
            depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})
            depther.cuda()
            depther.eval()
        elif opt.model_type == 'depthanything_v2' : # Depthanything 1 and 2 use the same architecture, so load the same structure, only diff weights
            depther = DepthAnythingV2(**{'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}) # only implemented for base model for now
            # depther = DepthAnythingV2(**{'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]})
            depther.load_state_dict(depther_dict)
            depther.cuda()
            depther.eval()
        elif opt.model_type == 'depthanything_v1':
            depther = DepthAnything({'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}) # only implemented for base model for now
            # depther = DepthAnythingV2(**{'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]})
            depther.load_state_dict(depther_dict)
            depther.cuda()
            depther.eval()
        elif opt.model_type == 'dino_v2':
            depther = model
            depther.cuda()
            depther.eval()
        elif opt.model_type == 'surgedepth':
            depther = SurgeDepth(**{'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]})
            depther.load_state_dict(depther_dict)
            depther.cuda()
            depther.eval()

        elif opt.model_type =='depthpro':
            depther = model
            depther.cuda()
            depther.eval()

    else:
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)
        if opt.eval_split == 'endovis':
            filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
            dataset = datasets.SCAREDRAWDataset(opt.data_path, filenames,
                                            opt.height, opt.width,
                                            [0], 4, is_train=False)

        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

    if opt.eval_split == 'endovis':
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
        
    if opt.visualize_depth:
        vis_dir = os.path.join(opt.vis_folder,opt.model_type, "vis_depth")
        os.makedirs(vis_dir, exist_ok=True)

    inference_times = []
    sequences = []
    keyframes = []
    frame_ids = []
    
    errors = []
    ratios = []
    maxs = []
    mins = []
    maxs_pred = []
    mins_pred = []
    mins_depth = []
    maxs_depth = []
    # pred_depths= np.array()
    index = 0
    nr_exceeds_max = {'nr': [],'abs_rel': [],'id':[]}
    print("-> Computing predictions with size {}x{}".format(
        opt.width, opt.height))

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader),total=len(dataloader)):
            if opt.eval_split == 'endovis':

                input_color = data[("color", 0, 0)].cuda()
                # input_color = transform({'image': input_color})['image']
                if opt.model_type == 'surgedepth':
                    input_color = torchvision.transforms.functional.normalize(input_color,mean=[0.46888983, 0.29536288, 0.28712815], std=[0.24689102 ,0.21034359, 0.21188641])
                elif opt.model_type == 'depthpro':
                    input_color = torchvision.transforms.functional.normalize(input_color,mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    input_color = input_color.to(torch.float16) # Depthpro requires half precision to run on laptop
                else:
                    input_color = torchvision.transforms.functional.normalize(input_color,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            elif opt.eval_split == 'endonerf' or opt.eval_split =='hamlyn':
                input_color = data['rgb'].cuda()
            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            if opt.ext_disp_to_eval is None:
                time_start = time.time()
                if opt.model_type == 'dino_v2':
                    output = depther.whole_inference(input_color,img_meta = None,rescale = True)
                    output_disp = output
                elif opt.model_type =='depthpro':
                    output = depther.infer(input_color)
                    output_disp = output['depth']
                else:
                    output = depther(input_color)

                # print(output.max())
                inference_time = time.time() - time_start
                if opt.model_type == 'endodac':
                    output_disp = output[("disp", 0)]
                elif opt.model_type == 'depthanything_v2' or opt.model_type =='depthanything_v1' or opt.model_type =='surgedepth':
                    output_disp = output
                

                if opt.model_type == 'depthanything_v1' or opt.model_type =='depthanything_v2' or opt.model_type =='surgedepth':
                    pred_disp = (output_disp-output_disp.min())/(output_disp.max()-output_disp.min())*9+1 # Scale all outputs between 1 and 10 for numerical stability
                else:
                    pred_disp,_ = disp_to_depth(output_disp, opt.min_depth, opt.max_depth)

                


                if opt.model_type =='endodac' or opt.model_type =='dino_v2':
                    pred_disp = pred_disp.cpu()[:, 0].numpy()
                elif opt.model_type == 'depthanything_v2' or opt.model_type == 'depthanything_v1' or opt.model_type =='surgedepth' or opt.model_type =='depthpro':
                    pred_disp = pred_disp.cpu().numpy()
                
                if not opt.model_type =='depthpro':
                    pred_disp = pred_disp[0] # should be WxH

                # plt.imshow(pred_disp)
                # plt.show()


            else:
                pred_disp = pred_disps[i]
                inference_time = 1
            inference_times.append(inference_time)
            
            if opt.eval_split == 'endovis': # For endovis, load gt from previously computed .npz depth maps            
                sequence = str(np.array(data['sequence'][0]))
                keyframe = str(np.array(data['keyframe'][0]))
                frame_id = "{:06d}.npz".format(data['frame_id'][0]-1)
                # print()
                gt_path = os.path.join(opt.data_path,'dataset_{}'.format(sequence),'keyframe_{}'.format(keyframe),'data/scene_points','depth{}'.format(frame_id))
                gt_depth = np.load(gt_path,fix_imports=True, encoding='latin1')["data"]
                
                
            elif opt.eval_split == 'endonerf': # Load GT from previously computed .npz depth maps
                frame_id ,_,_= data['id'][0].split('.')
                frame_id = "{}.npz".format(frame_id)
                gt_path = os.path.join(opt.data_path,'depth','depth_{}'.format(frame_id))
                gt_depth = np.load(gt_path,fix_imports=True, encoding='latin1')["data"]
            elif opt.eval_split =='hamlyn':
                gt_depth = data['depth'].squeeze().squeeze().numpy()
                


            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            if opt.model_type == 'dino_v2' or opt.model_type =='depthpro': # dino directly predicts depth, so no need to convert disp to depth
                pred_depth = pred_disp
            else:
                pred_depth = 1/pred_disp
            # if opt.model_type == 'depthanything_v1':
            #     pred_depth = rescale_linear(pred_depth,opt.min_depth,opt.max_depth)
            # plt.imshow(pred_depth)
            # plt.colorbar()
            # plt.show()
            
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            if opt.visualize_depth:
                if opt.eval_split =='endovis':
                    rgb = data[("color", 0, 0)].squeeze().permute(1,2,0)
                elif opt.eval_split =='endonerf' or opt.eval_split =='hamlyn':
                    rgb = data['rgb'].squeeze().permute(1,2,0)

                if opt.eval_split =='hamlyn':
                    frame_id ,_= data['id'][0].split('.')
                # vis_pred_depth = render_depth(pred_disp)
                fig,ax = plt.subplots(1,3,figsize = (15,5))
                ax[0].imshow(rgb)
                im1 = ax[1].imshow(pred_depth)
                plt.colorbar(im1,ax = ax[1])

                im2 = ax[2].imshow((gt_depth-gt_depth.min())/(gt_depth.max()-gt_depth.min()))
                plt.colorbar(im2,ax= ax[2])

                if opt.eval_split =='endovis':
                    vis_file_name = os.path.join(vis_dir, sequence + "_" +  keyframe + "_" + frame_id + ".png")
                elif opt.eval_split =='endonerf' or opt.eval_split =='hamlyn':
                    vis_file_name = os.path.join(vis_dir,frame_id + ".png")
                fig.savefig(vis_file_name)
                plt.close()
    
            # maxs_pred.append(pred_disp.max().item())
            # mins_pred.append(pred_disp.min().item()) 

            # print(gt_depth.shape)
            gt_depth = gt_depth[mask]
            # print(gt_depth.shape)
            # pred_disp = pred_disp[mask]

            if opt.model_type == 'depthanything_v1' or opt.model_type == 'depthanything_v2' or opt.model_type == 'surgedepth':
                gt_disp = 1/gt_depth
                if opt.model_type =='surgedepth':
                    pred_disp = 1/pred_depth
                pred_disp_aligned, t_gt, s_gt, t_pred, s_pred = align_shift_and_scale(gt_disp, pred_disp)
                # min_disp = 1/MAX_DEPTH
                # max_disp = 1/MIN_DEPTH
                # # print(f't_gt {t_gt}, s_gt {s_gt},t_pred {t_pred},s_pred{s_pred}')
                # pred_disp_aligned[pred_disp_aligned<min_disp] = min_disp
                # pred_disp_aligned[pred_disp_aligned>max_disp] = max_disp
                # maxs_depth.append(pred_disp_aligned.max())
                # mins_depth.append(pred_disp_aligned.min())

                pred_depth = 1 / pred_disp_aligned
            pred_depth = pred_depth[mask]

            if gt_depth.shape[0]<100: # If there are fewer than 1000 valid depth pixels, skip the current iteration
                counter+=1
                continue

            pred_depth *= opt.pred_depth_scale_factor
            # maxs_depth.append(pred_depth.max())
            # mins_depth.append(pred_depth.min())
            # np.append(pred_depths,pred_depth)
            # print(np.median(gt_depth))
            # print(np.median(pred_depth))
            # break
            if not opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                # print(ratio)
                if not np.isnan(ratio).all():
                    ratios.append(ratio)
                pred_depth *= ratio

            
            # maxs_depth.append(pred_depth.max())
            # mins_depth.append(pred_depth.min())
            # if pred_depth.max() >= MAX_DEPTH:
            #     counter+=1
            #     # nr_exceeds_max['nr'].append(sum(pred_depth>MAX_DEPTH))
            #     # index = 1
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            error = compute_errors(gt_depth, pred_depth)
            if error[0]>0.1:
                nr_exceeds_max['nr'].append(sum(pred_depth>MAX_DEPTH))
                nr_exceeds_max['abs_rel'].append(error[0])
                nr_exceeds_max['id'].append(i)
                # index = 0
                

            if not np.isnan(error).all():
                errors.append(error)

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    errors = np.array(errors)
    # maxs = np.array(maxs)
    # mins = np.array(mins)
    # maxs_pred = np.array(maxs_pred)
    # mins_pred = np.array(mins_pred)
    # mins_depth=np.array(mins_depth)
    # maxs_depth = np.array(maxs_depth)
    mean_errors = np.mean(errors, axis=0)
    # pred_depths = [depth for depth_list in pred_depths for depth in depth_list]
    # print(len(pred_depths))
    cls = []
    for i in range(len(mean_errors)):
        cl = st.t.interval(confidence=0.95, df=len(errors)-1, loc=mean_errors[i], scale=st.sem(errors[:,i]))
        cls.append(cl[0])
        cls.append(cl[1])
    cls = np.array(cls)
    print("\n       " + ("{:>11}      | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print("mean:" + ("&{: 12.3f}      " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("cls: " + ("& [{: 6.3f}, {: 6.3f}] " * 7).format(*cls.tolist()) + "\\\\")
    print("average inference time: {:0.1f} ms".format(np.mean(np.array(inference_times))*1000))
    # print('Average max prediction {}, avg min prediction {}'.format(maxs.max(),mins.min()))
    # print('Average max prediction after norm {}, avg min prediction {}'.format(maxs_pred.max(),mins_pred.min()))
    # print('Average max depth prediction after norm {}, avg min prediction {}'.format(maxs_depth.max(),mins_depth.min()))
    # print('Nr exceeds {}'.format(nr_exceeds_max))
    print("\n-> Done!")
    # print('counter: {},total recorded errors {}'.format(counter,len(errors)))

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
