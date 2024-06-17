import os, shutil
import argparse
from PIL import Image, ImageDraw as D
import torchvision
from util.func import get_patch_size,get_device, load_net, get_project_loader
from torchvision import transforms
import torch
from util.vis_pipnet import get_img_coordinates
from util.args import get_args
import json


def vis_pred(net, vis_test_dir, classes, device, args: argparse.Namespace):
    # Make sure the model is in evaluation mode
    net.eval()

    save_dir = os.path.join(args.log_dir, args.dir_for_saving_images)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    num_workers = args.num_workers

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(args.image_size, args.image_size)),
                            transforms.ToTensor(),
                            normalize])

    vis_test_set = torchvision.datasets.ImageFolder(vis_test_dir, transform=transform_no_augment)
    vis_test_loader = torch.utils.data.DataLoader(vis_test_set, batch_size = 1,
                                                shuffle=False, pin_memory=not args.disable_cuda and torch.cuda.is_available(),
                                                num_workers=num_workers)
    imgs = vis_test_set.imgs
    
    last_y = -1
    
    for k, (xs, ys) in enumerate(vis_test_loader): #shuffle is false so should lead to same order as in imgs
        if ys[0] != last_y:
            last_y = ys[0]
            count_per_y = 0
        else:
            count_per_y +=1
            #if count_per_y>5: #show max 5 imgs per class to speed up the process
            #    continue
        xs, ys = xs.to(device), ys.to(device)
        img = imgs[k][0]
        img_name = os.path.splitext(os.path.basename(img))[0]
        img_name_dict = {}
        dir = os.path.join(save_dir,img_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
            shutil.copy(img, dir)
        
        with torch.no_grad():
            softmaxes, pooled, out = net(xs, inference=True) #softmaxes has shape (bs, num_prototypes, W, H), pooled has shape (bs, num_prototypes), out has shape (bs, num_classes)
            wshape = softmaxes.shape[-1]
            args.wshape = wshape 
            patchsize, skip = get_patch_size(args)
            sorted_out, sorted_out_indices = torch.sort(out.squeeze(0), descending=True)
            for pred_class_idx in sorted_out_indices[:3]:
                pred_class = classes[pred_class_idx]
                img_name_dict[pred_class] = {}
                save_path = os.path.join(dir, pred_class+"_"+str(f"{out[0,pred_class_idx].item():.3f}"))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                sorted_pooled, sorted_pooled_indices = torch.sort(pooled.squeeze(0), descending=True)
                simweights = []
                prototype_ids = []
                similarities = []
                weights = []
                h_coor_mins = []
                h_coor_maxs = []
                w_coor_mins = []
                w_coor_maxs = []
                for prototype_idx in sorted_pooled_indices:
                    simweight = pooled[0,prototype_idx].item() * net.module._classification.weight[pred_class_idx, prototype_idx].item()
                    simweights.append(simweight)
                    if abs(simweight) > 0.01:
                        max_h, max_idx_h = torch.max(softmaxes[0, prototype_idx, :, :], dim=0)
                        max_w, max_idx_w = torch.max(max_h, dim=0)
                        max_idx_h = max_idx_h[max_idx_w].item()
                        max_idx_w = max_idx_w.item()
                        image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img))
                        img_tensor = transforms.ToTensor()(image).unsqueeze_(0) #shape (1, 3, h, w)
                        h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, softmaxes.shape, patchsize, skip, max_idx_h, max_idx_w)
                        img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                        img_patch = transforms.ToPILImage()(img_tensor_patch)
                        img_patch.save(os.path.join(save_path, 'mul%s_p%s_sim%s_w%s_patch.png'%(str(f"{simweight:.3f}"),str(prototype_idx.item()),str(f"{pooled[0,prototype_idx].item():.3f}"),str(f"{net.module._classification.weight[pred_class_idx, prototype_idx].item():.3f}"))))
                        draw = D.Draw(image)
                        draw.rectangle([(max_idx_w*skip,max_idx_h*skip), (min(args.image_size, max_idx_w*skip+patchsize), min(args.image_size, max_idx_h*skip+patchsize))], outline='yellow', width=2)
                        image.save(os.path.join(save_path, 'mul%s_p%s_sim%s_w%s_rect.png'%(str(f"{simweight:.3f}"),str(prototype_idx.item()),str(f"{pooled[0,prototype_idx].item():.3f}"),str(f"{net.module._classification.weight[pred_class_idx, prototype_idx].item():.3f}"))))
                        prototype_ids.append(prototype_idx.item())
                        similarities.append(pooled[0,prototype_idx].item())
                        weights.append(net.module._classification.weight[pred_class_idx, prototype_idx].item())
                        h_coor_mins.append(max_idx_h*skip)
                        h_coor_maxs.append(min(args.image_size, max_idx_h*skip+patchsize))
                        w_coor_mins.append(max_idx_w*skip)
                        w_coor_maxs.append(min(args.image_size, max_idx_w*skip+patchsize))
                img_name_dict[pred_class]["simweights"] = [i for i in simweights if i >= 0.01]
                img_name_dict[pred_class]["prototype_ids"] = prototype_ids
                img_name_dict[pred_class]["similarity"] = similarities
                img_name_dict[pred_class]["weights"] = weights
                img_name_dict[pred_class]["h_coor_min"] = h_coor_mins
                img_name_dict[pred_class]["h_coor_max"] = h_coor_maxs
                img_name_dict[pred_class]["w_coor_min"] = w_coor_mins
                img_name_dict[pred_class]["w_coor_max"] = w_coor_maxs
        with open(f"{args.log_dir}{img_name}.json", "w") as outfile:
            json.dump(img_name_dict, outfile)
            
if __name__=="__main__":

    global device 
    device = get_device()
    args = get_args() # gets the standard arguments, please check if they apply to your case
    model_dir = "D:/Joan/PIPNet-deployment-main/model/" # this is the folder you get the model from
    args.log_dir = "D:/Joan/PIPNet-deployment-main/test_results" # this is the folder you save the results to
    net = load_net(f"{model_dir}checkpoints/net_trained_last") # the location of the net
    classes = ["frac", "no_frac"] # also standard classes, might neeed to be adjusted
    vis_test_dir = "D:/Joan/PIPNet-deployment-main/pukkaJ-in/" # this is the folder with the images you want to run the inference over


    vis_pred(net, vis_test_dir, classes, device, args)