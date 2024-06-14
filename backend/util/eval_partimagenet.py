import torch
from tqdm import tqdm 
import json
from util.func import get_patch_size
from PIL import Image
import os
import numpy as np
from util.vis_pipnet import get_img_coordinates

@torch.no_grad()                    
def eval_prototypes_partimagenet_annotations(net, projectloader, json_path, epoch, device, log, args, pretrain=False):
    category_id_to_name = dict()
    category_id_to_supercategory = dict()
    with open(json_path) as f:
        annotations = json.load(f)

        for catjson in annotations['categories']:
            category_id_to_name[catjson['id']]=catjson['name']
            category_id_to_supercategory[catjson['id']]=catjson['supercategory']
        print(category_id_to_name, flush=True)
    
        # Make sure the model is in evaluation mode
        net.eval()
        classification_weights = net.module._classification.weight
        patchsize, skip = get_patch_size(args)

        # Show progress on progress bar
        project_iter = tqdm(enumerate(projectloader),
                            total=len(projectloader),
                            desc='Evaluating Prototypes PartImageNet parts',
                            mininterval=50.,
                            ncols=0)
        imgs = projectloader.dataset.imgs

        proto_parts_overlaps = dict()
        num_images_with_active_prototype = dict()
        
        images_seen = 0
        # Iterate through the training set
        for i, (xs, ys) in project_iter:
            
            images_seen+=1
            xs, ys = xs.to(device), ys.to(device)

            with torch.no_grad():
                img_path = imgs[i][0]
                img = Image.open(img_path)
                img_orig_width, img_orig_height = img.size

                # Use the model to classify this batch of input data
                pfs, pooled, _ = net(xs,inference=True)
                pooled = pooled.squeeze(0) #shape(2048)
                pfs = pfs.squeeze(0) #shape (2048,24,24)
                _, filename = os.path.split(img_path)
                
                for imgjson in annotations['images']:
                    if imgjson['file_name'] == filename:
                        img_id = imgjson['id']
                        
                        for p in range(pooled.shape[0]):
                            c_weight = torch.max(classification_weights[:,p]) 
                            if c_weight > 1e-3 or pretrain:#ignore prototypes that are not relevant to any class
                                if pooled[p]>0.5: #prototype is active. 
                                    if p not in num_images_with_active_prototype:
                                        num_images_with_active_prototype[p] = 0
                                    num_images_with_active_prototype[p] += 1
                                    
                                    location_h, location_h_idx = torch.max(pfs[p,:,:], dim=0) #analyse single location in image where prototype is most active.
                                    _, location_w_idx = torch.max(location_h, dim=0)
                                    location = (location_h_idx[location_w_idx].item(), location_w_idx.item())
                                    resized_img_location_h_min, resized_img_location_h_max, resized_img_location_w_min, resized_img_location_w_max = get_img_coordinates(args.image_size, pfs.shape, patchsize, skip, location[0], location[1]) 
 
                                    # resize to original image size 
                                    orig_img_location_h_min = (img_orig_height/args.image_size)*resized_img_location_h_min
                                    orig_img_location_h_max = (img_orig_height/args.image_size)*resized_img_location_h_max
                                    orig_img_location_w_min = (img_orig_width/args.image_size)*resized_img_location_w_min
                                    orig_img_location_w_max = (img_orig_width/args.image_size)*resized_img_location_w_max
                                    
                                    most_overlap = dict()
                                    for annojson in annotations['annotations']:
                                        if annojson['image_id']==img_id:
                                            if p not in proto_parts_overlaps:
                                                proto_parts_overlaps[p]=dict()
                                            x_gt,y_gt,width_gt,height_gt = annojson['bbox']
                                            category_id = annojson['category_id']
                                            if category_id not in most_overlap:
                                                most_overlap[category_id] = 0.
                                            if category_id not in proto_parts_overlaps[p]:
                                                proto_parts_overlaps[p][category_id]=[]

                                            # calculate overlap https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
                                            # determine the (x, y)-coordinates of the intersection rectangle
                                            xA = max(orig_img_location_w_min, x_gt)
                                            yA = max(orig_img_location_h_min, y_gt)
                                            xB = min(orig_img_location_w_max, x_gt+width_gt)
                                            yB = min(orig_img_location_h_max, y_gt+height_gt)
                                            # compute the area of intersection rectangle
                                            overlap = max(0., xB - xA + 1.) * max(0., yB - yA + 1.)
                                            # overlap = overlap / annojson['area']
                                            if annojson['area'] == 0.:
                                                overlap = overlap / (patchsize*patchsize)
                                            else:
                                                overlap = overlap / min(annojson['area'], patchsize*patchsize)
                                
                                            if overlap >= most_overlap[category_id]:
                                                most_overlap[category_id] = overlap
                                    for cid in most_overlap:
                                        proto_parts_overlaps[p][cid].append(most_overlap[cid])
                
    print("\n Number of prototypes with activation > 0.5: ", len(proto_parts_overlaps.keys()), list(proto_parts_overlaps.keys()), "\n", flush=True)
    purities = dict()
    best_purities = dict()
    best_category_purities = dict()
    overall_purity_most_occuring_category = dict ()
    supcat_overlap_ratios = dict()
    sum_binarized_overlaps_per_prototype_per_category = dict()
    num_prototypes_without_any_overlap = 0
    log.log_values('log_epoch_overview', "p_partimagenet", "mean overall purity", "std overall purity", "mean most purity", "std most purity", "mean best purity", "mean best supercategory overlap ratio", "# prototypes activation > 0.5", "#prototypes without any overlap")
    for threshold in [0.001]:
        for proto in proto_parts_overlaps:
            proto_has_overlap = False
            if proto not in sum_binarized_overlaps_per_prototype_per_category:
                sum_binarized_overlaps_per_prototype_per_category[proto] = dict()
            if proto not in purities:
                purities[proto] = dict()
            best_purity = 0.
            proto_presence = num_images_with_active_prototype[proto]
            presence_overlap_ratio = dict()
            for category in proto_parts_overlaps[proto]:
                overlaps = proto_parts_overlaps[proto][category]
                binarized_overlaps = np.where(np.array(overlaps) > threshold, 1, 0)
                sum_binarized_overlaps_per_prototype_per_category[proto][category] = binarized_overlaps.sum()
                purity = np.mean(binarized_overlaps)
                presence_overlap_ratio[category]=binarized_overlaps.sum()/proto_presence
                if max(overlaps)>0.: #do not calculate purity for category where there is zero overlap to prototype
                    proto_has_overlap = True
                    purities[proto][category] = purity
                    if purity > best_purity:
                        best_purity = purity
            if not proto_has_overlap:
                num_prototypes_without_any_overlap += 1
            if best_purity > 0.:
                best_purities[proto] = best_purity
            sum_binarized_overlaps_supercategory = dict()
            presence_overlap_ratio_supcat = dict()
            for category in proto_parts_overlaps[proto]:
                supcat = category_id_to_supercategory[category]
                if supcat not in sum_binarized_overlaps_supercategory:
                    sum_binarized_overlaps_supercategory[supcat] = 0
                sum_binarized_overlaps_supercategory[supcat]+=sum_binarized_overlaps_per_prototype_per_category[proto][category]
            for supcat in sum_binarized_overlaps_supercategory:
                presence_overlap_ratio_supcat[supcat] = sum_binarized_overlaps_supercategory[supcat] / proto_presence
            best_ratio_supercategory = max(presence_overlap_ratio_supcat, key=presence_overlap_ratio_supcat.get)
            supcat_overlap_ratios[proto] = presence_overlap_ratio_supcat[best_ratio_supercategory]
            

            # get category with most often overlap with prototype. 
            if proto_has_overlap:
                best_ratio_category = max(presence_overlap_ratio, key=presence_overlap_ratio.get)
                if best_ratio_category in purities[proto]:
                    best_category_purities[proto] = purities[proto][best_ratio_category]
                if proto in num_images_with_active_prototype and num_images_with_active_prototype[proto] > 0:
                    overall_purity_most_occuring_category[proto] = sum_binarized_overlaps_per_prototype_per_category[proto][best_ratio_category] / proto_presence
                
                if purities[proto]:
                    print("Prototype", proto, "has purities:", purities[proto], "with occurences", sum_binarized_overlaps_per_prototype_per_category[proto], flush=True)
        
        mean_best_purity = np.mean(list(best_purities.values()))
        mean_most_purity = np.mean(list(best_category_purities.values()))
        overall_purity = np.mean(list(overall_purity_most_occuring_category.values()))
        print("Overall purity averaged over all active non-zero weighted prototypes:", overall_purity, "mean most purity:", mean_most_purity, "mean best purity: ", mean_best_purity, flush=True)

        
        log.log_values('log_epoch_overview', "p_partimagenet", overall_purity, np.std(list(overall_purity_most_occuring_category.values())), mean_most_purity, np.std(list(best_category_purities.values())), mean_best_purity, np.mean(list(supcat_overlap_ratios.values())), len(proto_parts_overlaps.keys()), num_prototypes_without_any_overlap)  
