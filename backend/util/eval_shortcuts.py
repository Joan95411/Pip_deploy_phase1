import pandas as pd
import numpy as np
import torch
from PIL import Image
from util.func import get_patch_size, get_project_loader, get_comparison_project_loader
from util.vis_pipnet import get_img_coordinates
import re
from tqdm import tqdm
from collections import Counter

pd.set_option('mode.chained_assignment', None)

def get_prototype_positions(net, args, projectloader, device):

    """
    Get the most activated position of the relevant prototypes per image
    """

    #projectloader = get_project_loader(vis_dir=vis_dir, args=args)
    imgs = projectloader.dataset.imgs
    net.eval()
    classification_weights = net.module._classification.weight
    #img_iter = enumerate(projectloader)
    img_iter = tqdm(enumerate(projectloader),
                        total=len(projectloader),
                        desc="prototype position calculation",
                        mininterval=5.,
                        ncols=0)
    (xs, ys) = next(iter(projectloader))
    h_coor_min_list, h_coor_max_list, w_coor_min_list, w_coor_max_list, img_names, prototypes = [], [], [], [], [], []
    for i, (xs, ys) in img_iter:
        #print(i)
        xs, ys = xs.to(device), ys.to(device)
        img_path = imgs[i][0]
        # print(img_path)
        img = Image.open(img_path)
        img_orig_width, img_orig_height = img.size
        # Use the model to classify this batch of input data
        with torch.no_grad():
            softmaxes, pooled, out = net(xs, inference=True) 
            pooled = pooled.squeeze(0)
            softmaxes = softmaxes.squeeze(0)
            wshape = softmaxes.shape[-1]
            args.wshape = wshape 
        patchsize, skip = get_patch_size(args)
        for p in range(pooled.shape[0]):
            c_weight = torch.max(classification_weights[:,p]) #ignore prototypes that are not relevant to any class
            if c_weight>1e-3:
                if pooled[p]>0.1: #prototype is active. 
                    location_h, location_h_idx = torch.max(softmaxes[p,:,:], dim=0) #analyse single location in image where prototype is most active.
                    _, location_w_idx = torch.max(location_h, dim=0)
                    location = (location_h_idx[location_w_idx].item(), location_w_idx.item())
                    resized_img_location_h_min, resized_img_location_h_max, resized_img_location_w_min, resized_img_location_w_max = get_img_coordinates(args.image_size, softmaxes.shape, patchsize, skip, location[0], location[1]) 
                       
    
                    # resize to original image size 
                    orig_img_location_h_min = (img_orig_height/args.image_size)*resized_img_location_h_min
                    orig_img_location_h_max = (img_orig_height/args.image_size)*resized_img_location_h_max
                    orig_img_location_w_min = (img_orig_width/args.image_size)*resized_img_location_w_min
                    orig_img_location_w_max = (img_orig_width/args.image_size)*resized_img_location_w_max

                    h_coor_min_list.append(orig_img_location_h_min)
                    h_coor_max_list.append(orig_img_location_h_max)
                    w_coor_min_list.append(orig_img_location_w_min)
                    w_coor_max_list.append(orig_img_location_w_max)
                    
                    # append image name 
                    regex = "([^\/]+$)"
                    img_name = re.findall(regex, img_path)
                    img_name = img_name[0].replace("_rect", "")
                    img_names.append(img_name)
                    prototypes.append(p)
    return h_coor_min_list, h_coor_max_list, w_coor_min_list, w_coor_max_list, img_names, prototypes


def locate_prototypes(net, args, projectloader, device, file_name):
    """
    Compare the position of the most activated prototypes per image to the postitions of the shortcut rectangles
    """
    vis_dir = args.vis_dir
    csv_name = args.csv_name
    log_dir = args.log_dir

    rect_coordinates = pd.read_csv(f"{vis_dir}/{csv_name}")
    rect_coordinates['prototype'] = 0
    rect_coordinates["overlap"] = 0
    rect_coordinates["iou"] = 0
    rect_coordinates["proto_w_min"] = 0
    rect_coordinates["proto_w_max"] = 0
    rect_coordinates["proto_h_min"] = 0
    rect_coordinates["proto_h_max"] = 0

    h_coor_min_list, h_coor_max_list, w_coor_min_list, w_coor_max_list, img_names, prototypes = get_prototype_positions(net, args, projectloader, device)
    assert len(img_names) == len(prototypes)
    img_name_cache = ""

    img_iter = tqdm(enumerate(img_names),
                        total=len(img_names),
                        desc="prototype location",
                        mininterval=5.,
                        ncols=0)
    img_name = next(iter(img_names))

    for i, img_name in img_iter:
        img_specific_coordinates = rect_coordinates[rect_coordinates.filename.str.contains(img_name[:-4])]

        if img_specific_coordinates.empty is False:   
            #print(i) 
            x_gt,y_gt,width_gt,height_gt = img_specific_coordinates["start_i"].values[0], img_specific_coordinates["start_j"].values[0], img_specific_coordinates["w"].values[0], img_specific_coordinates["h"].values[0] 
            xA = max(w_coor_min_list[i], x_gt)
            yA = max(h_coor_min_list[i], y_gt)
            xB = min(w_coor_max_list[i], x_gt+width_gt)
            yB = min(h_coor_max_list[i], y_gt+height_gt)
            overlap = max(0., xB - xA) * max(0., yB - yA)
            prototype_area = (w_coor_max_list[i] - w_coor_min_list[i]) *  (h_coor_max_list[i] - h_coor_min_list[i])
            shortcut_area = width_gt * height_gt
            correction = prototype_area/shortcut_area # correct for the size difference between prototype and shortcut 
            iou = overlap / float(prototype_area + shortcut_area - overlap) * correction
            prototype = prototypes[i]

            if img_name == img_name_cache:
                new_row = ({"filename": img_name, "w":width_gt, "h":height_gt, "start_i":x_gt, "start_j":y_gt, "overlap": overlap, "iou": iou, "prototype": prototype,
                            "proto_w_min":w_coor_min_list[i], "proto_w_max": w_coor_max_list[i], "proto_h_min":h_coor_min_list[i], "proto_h_max":h_coor_max_list[i]})
                rect_coordinates = rect_coordinates.append(new_row, ignore_index=True)
            else: 
                rect_coordinates["overlap"][rect_coordinates.filename.str.contains(img_name[:-4])] = overlap
                rect_coordinates["iou"][rect_coordinates.filename.str.contains(img_name[:-4])] = iou
                rect_coordinates["prototype"][rect_coordinates.filename.str.contains(img_name[:-4])] = prototype
                rect_coordinates["proto_w_min"][rect_coordinates.filename.str.contains(img_name[:-4])] = w_coor_min_list[i]
                rect_coordinates["proto_w_max"][rect_coordinates.filename.str.contains(img_name[:-4])] = w_coor_max_list[i]
                rect_coordinates["proto_h_min"][rect_coordinates.filename.str.contains(img_name[:-4])] = h_coor_min_list[i]
                rect_coordinates["proto_h_max"][rect_coordinates.filename.str.contains(img_name[:-4])] = h_coor_max_list[i]

            img_name_cache = img_name
    rect_coordinates.to_csv(f"{log_dir}/{file_name}.csv")
    grouped_df = rect_coordinates.groupby('prototype').mean()
    grouped_df2 = rect_coordinates.groupby('prototype').count()
    # only take those with a minimum overlap of 0.1
    shortcut_prototype_df = grouped_df[grouped_df["iou"]>0.1].join(grouped_df2,on="prototype", rsuffix="_r")[["iou", "iou_r"]]
    index_list = shortcut_prototype_df.index.tolist()
    return index_list



def evaluate_shortcut_importance(net, args, device, file_name, projectloader_with_rectangles, projectloader_without_rectangles):
    vis_dir = args.vis_dir
    csv_name = args.csv_name
    log_dir = args.log_dir

    rect_coordinates = pd.read_csv(f"{vis_dir}/{csv_name}", index_col=0)

    net.eval()
    #img_iter = enumerate(zip(projectloader_with_rectangles, projectloader_without_rectangles))
    projectloader_zip = zip(projectloader_with_rectangles, projectloader_without_rectangles)
    img_iter = tqdm(enumerate(projectloader_zip),
                        total=len(projectloader_without_rectangles),
                        desc="shortcut importance evaluation",
                        mininterval=5.,
                        ncols=0)
    ((xs_with_rectangles, ys_with_rectangles), (xs_without_rectangles, ys_without_rectangles)) = next(iter(projectloader_zip))

    out1s = list()
    out2s = list()
    weighted_differences = list()
    index_list = list()
    changed_prediction = 0
    imgs = projectloader_with_rectangles.dataset.imgs
    
    rect_coordinates["sum_of_change_prototypes"] = 0
    rect_coordinates["percentage_of_full_prediction"] = 0
    rect_coordinates["out_with_rectangle"] = 0
    rect_coordinates["out_without_rectangle"] = 0
    rect_coordinates["most_changed_prototype_indices"] = 0
    rect_coordinates["changed_prediction"] = "False"
    
    for i, ((xs_with_rectangles, ys_with_rectangles), (xs_without_rectangles, ys_without_rectangles)) in img_iter:
        xs_with_rectangles, ys_with_rectangles = xs_with_rectangles.to(device), ys_with_rectangles.to(device)
        xs_without_rectangles, ys_without_rectangles = xs_without_rectangles.to(device), ys_without_rectangles.to(device)

        img_path = imgs[i][0]

        # append image name 
        regex = "([^\/]+$)"
        img_name = re.findall(regex, img_path)
        img_name = img_name[0].replace("_rect", "")
        #print(img_name)
        # Use the model to classify this batch of input data
        with torch.no_grad():
            _, pooled1, out1 = net(xs_with_rectangles, inference=True) 
            _, pooled2, out2 = net(xs_without_rectangles, inference=True) 
            #print(i)
            proto_weights = net.module._classification.weight[1,:]
            if not torch.equal(out1, out2):
                weighted_pooled1 = torch.mul(pooled1, proto_weights)
                weighted_pooled2 = torch.mul(pooled2, proto_weights)

                difference = weighted_pooled1-weighted_pooled2
                no_of_changed_prototypes = torch.sum(torch.where(difference>1,1,0))

                most_changed_weighted_prototypes, indices = torch.topk(difference, no_of_changed_prototypes) 
                
                out1s.extend(out1.cpu().numpy())
                out2s.extend(out2.cpu().numpy())

                weighted_differences.extend(most_changed_weighted_prototypes.cpu()[0].numpy())
                index_list.extend(indices.cpu()[0].numpy())
                
                sum_of_changed_prototypes = torch.sum(most_changed_weighted_prototypes).cpu().item()
                percentage_of_full_prediction = torch.sum(most_changed_weighted_prototypes).cpu().numpy()/out1.cpu().numpy()[0][1].item()
                rect_coordinates["sum_of_change_prototypes"][rect_coordinates.filename.str.contains(img_name[:-4])] = sum_of_changed_prototypes
                rect_coordinates["percentage_of_full_prediction"][rect_coordinates.filename.str.contains(img_name[:-4])] = percentage_of_full_prediction
                rect_coordinates["out_with_rectangle"][rect_coordinates.filename.str.contains(img_name[:-4])] = np.array2string(out1.cpu().numpy()[0])
                rect_coordinates["out_without_rectangle"][rect_coordinates.filename.str.contains(img_name[:-4])] = np.array2string(out2.cpu().numpy()[0])
                rect_coordinates["most_changed_prototype_indices"][rect_coordinates.filename.str.contains(img_name[:-4])] = np.array2string(indices.cpu()[0].numpy())
                if np.argmax(out1.cpu().numpy()) != np.argmax(out2.cpu().numpy()):
                    changed_prediction +=1
                    rect_coordinates["changed_prediction"][rect_coordinates.filename.str.contains(img_name[:-4])] = "True"

    rect_coordinates.to_csv(f"{log_dir}/{file_name}.csv")
    return index_list

def adjustable_overlap_calculation(number, net, args, device, iou_filename, shortcuts_importance_filename):
    projectloader = get_project_loader(vis_dir=args.rect_dir, args=args, number=number)
    index_list_iou = locate_prototypes(net, args, projectloader, device, iou_filename)
    projectloader_with_rectangles, projectloader_without_rectangles = get_comparison_project_loader(vis_dir=args.rect_dir, norm_dir=args.norm_dir, args=args, number=number)
    index_list_importance = evaluate_shortcut_importance(net, args, device, shortcuts_importance_filename,projectloader_with_rectangles, projectloader_without_rectangles)
    counter = Counter(index_list_importance)
    overlap_counter = counter & Counter(index_list_iou)
    overlap = list(overlap_counter.keys())
    return overlap