import random
import torch
import os
import torch.nn.functional as F
from torch import nn
import numpy as np

def KL_loss(preds_now,preds_origin):
    loss=F.kl_div( F.log_softmax( preds_now, dim=-1),  F.log_softmax(preds_origin, dim=-1),  reduction='sum',  log_target=True ) * (1 * 1) / preds_origin.numel()
    return loss

def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        if param.requires_grad:
            params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)  
   

def ECE(conf, pred, gt, conf_bin_num = 10):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  
        
    Returns:
        ece: expected calibration error
    """
    bins = np.linspace(0, 1, conf_bin_num+1)
    bin_indices = np.digitize(conf, bins) - 1

    bin_acc = []
    bin_confidences = []
    for i in range(conf_bin_num):

        in_bin = bin_indices == i

        if np.sum(in_bin) > 0:
            accuracy = np.mean(gt[in_bin] == pred[in_bin])
            mean_confidence = np.mean(conf[in_bin])
        else:
            accuracy = 0
            mean_confidence = 0
        bin_acc.append(accuracy)
        bin_confidences.append(mean_confidence)


    bin_acc = np.array(bin_acc)
    bin_confidences = np.array(bin_confidences)


    weights = np.histogram(conf, bins)[0] / len(conf)
    ece = np.sum(weights * np.abs(bin_confidences - bin_acc))
        
    return ece

def save_gradient_ratio(forget_trainloader, model, optimizer,save_dir):
    gradients = {}
    forget_loader = forget_trainloader
    model.eval()
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradients[name]= torch.zeros_like(param).cuda()
    for i, (image, target_coarse,target_fine,_) in enumerate(forget_loader):
        image = image.cuda()
        target_fine = target_fine.cuda()
        target_coarse = target_coarse.cuda()
        # compute output
        output_coarse,output_fine,_ = model(image,training=True)
        loss = - F.cross_entropy(output_fine, target_fine)
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data
    

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])
    

    threshold_list = [0.001,0.01,0.1]

    for i in threshold_list:
        sorted_dict_positions = {}
        hard_dict = {}
        all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])
        threshold_index = int(len(all_elements) * i)
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)
        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            tensor_ranks = ranks[start_index : start_index + num_elements]
            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements
        torch.save(hard_dict, os.path.join(save_dir, "difficult_with_{}.pt".format(i)))


def find_random_fine_models(model_names,fine_classes):
    results = []
    for model_name in model_names:
        results.append(random.choice(fine_classes))
    return results

def find_random_coarse_models(model_names):
    results = []
    for model_name in model_names:
        for models in list(car_dict.keys()):
            if model_name in models:
                    results.append(random.choice(list(car_dict.keys())))
                    break
    return results
