import os
from utils import *
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from .utils_helper import*
import copy
from collections import OrderedDict
import wandb
import json
from tqdm import tqdm

def Unlearn_training(model, zeroshot_model, trainloader, testloader, optimizer, scheduler, epoch, args, log_interval, iterations,SALUN_mask):
    tl = Averager()
    ta_coarse = Averager()
    ta_fine = Averager()
    model = model.train()
    zeroshot_model = zeroshot_model.eval()
    tqdm_gen = tqdm(trainloader, desc='Unlearn Dataset Training')
    for UL_batch in tqdm_gen:
        iterations += 1
        # Load data
        data, coarse_labels, fine_labels, path = [_ for _ in UL_batch]
        data = data.cuda(non_blocking=True)
        coarse_labels = coarse_labels.cuda(non_blocking=True)
        fine_labels = fine_labels.cuda(non_blocking=True)
        # forward process
        preds_coarse, preds_fine, images_features = model(data, training=True)
        origin_coarse, origin_fine, zeroshot_features = zeroshot_model(data)
        #Compute for fine-grained and coarse-grained loss 
        loss_coarse = F.cross_entropy(preds_coarse, coarse_labels)
        loss_fine = F.cross_entropy(preds_fine, fine_labels)
        #logits for unlearned fine classes
        logits_fine_masked = torch.stack([torch.cat([logits[:label], logits[label + 1:]]) for logits, label in zip(preds_fine, fine_labels)])
        origin_fine_masked = torch.stack([torch.cat([logits[:label], logits[label + 1:]]) for logits, label in zip(origin_fine, fine_labels)])
        logits_coarse_masked = torch.stack([torch.cat([logits[:label], logits[label + 1:]]) for logits, label in zip(preds_coarse, coarse_labels)])
        origin_coarse_masked = torch.stack([torch.cat([logits[:label], logits[label + 1:]]) for logits, label in zip(origin_coarse, coarse_labels)])
        if args.unlearning_method=='Hinge':
            if args.unlearning_goal == 'fine':
                margin_fine = args.margin_f  
                logits_fine = preds_fine  
                batch_size, num_classes = logits_fine.size()
                mask = torch.ones_like(logits_fine).scatter_(1, fine_labels.unsqueeze(1), 0)
                masked_logits = logits_fine * mask
                max_incorrect_logits, _ = masked_logits.max(dim=1)
                correct_logits = logits_fine[torch.arange(batch_size), fine_labels]
                hinge_loss_fine = torch.clamp(correct_logits - max_incorrect_logits + margin_fine, min=0).mean()
                loss =hinge_loss_fine+ KL_loss(preds_coarse, origin_coarse) * args.KL_c + KL_loss(logits_fine_masked, origin_fine_masked) * args.KL_f  # Added kl_fine_loss
            else:
                margin_fine,marigin_coarse = args.margin_f,args.margin_c
                logits_fine,logits_coarse = preds_fine,preds_coarse  
                batch_size, num_fine_classes = logits_fine.size()
                _,num_coarse_classes=logits_coarse.size()
                mask_fine = torch.ones_like(logits_fine).scatter_(1, fine_labels.unsqueeze(1), 0)
                mask_coarse = torch.ones_like(logits_coarse).scatter_(1, coarse_labels.unsqueeze(1), 0)
                masked_logits_fine = logits_fine * mask_fine
                masked_logits_coarse = logits_coarse * mask_coarse
                max_incorrect_logits_fine, _ = masked_logits_fine.max(dim=1)
                max_incorrect_logits_coarse, _ = masked_logits_coarse.max(dim=1)
                correct_logits_fine = logits_fine[torch.arange(batch_size), fine_labels]
                correct_logits_coarse = logits_coarse[torch.arange(batch_size), coarse_labels]
                hinge_loss_fine = torch.clamp(correct_logits_fine - max_incorrect_logits_fine + margin_fine, min=0).mean()
                hinge_loss_coarse = torch.clamp(correct_logits_coarse - max_incorrect_logits_coarse + margin_coarse, min=0).mean()
                loss =hinge_loss_fine+hinge_loss_coarse+ KL_loss(logits_coarse_masked, origin_coarse_masked) * args.KL_c + KL_loss(logits_fine_masked, origin_fine_masked) * args.KL_f
        elif args.unlearning_method=='GDiff':
            if args.unlearning_goal == 'fine':
                loss =-loss_fine+ loss_coarse
            else:
                loss = -loss_fine - loss_coarse
        elif args.unlearning_method=='NPO':
            if args.unlearning_goal=='fine':
                loss_ref_fine=F.cross_entropy(origin_fine, fine_labels)
                loss_current_fine=loss_fine
                neg_log_ratios_fine = loss_current_fine - loss_ref_fine
                loss = -F.logsigmoid(args.beta * (neg_log_ratios_fine)).mean() * 2 / args.beta+KL_loss(preds_coarse , origin_coarse)*args.KL_c+args.KL_f*KL_loss(logits_fine_masked, origin_fine_masked)#loss_coarse*0.3
            else:
                neg_log_coarse = loss_coarse - F.cross_entropy(origin_coarse, coarse_labels)
                neg_log_fine=loss_fine-F.cross_entropy(origin_fine, fine_labels)
                loss = -F.logsigmoid(args.beta * (neg_log_coarse+neg_log_fine)).mean() * 2 / args.beta
        elif args.unlearning_method=='KL':
            if args.unlearning_goal == 'fine':
                loss =-loss_fine+ KL_loss(preds_coarse, origin_coarse) * args.KL_c + KL_loss(logits_fine_masked, origin_fine_masked) * args.KL_f  # Added kl_fine_loss
            else:
                loss =-loss_fine-loss_coarse+ KL_loss(logits_coarse_masked, origin_coarse_masked) * args.KL_c + KL_loss(logits_fine_masked, origin_fine_masked) * args.KL_f
        elif args.unlearning_method=='ME':
            if args.unlearning_goal == 'fine':
                num_labels = preds_fine.shape[-1]
                soft_outputs = F.softmax(preds_fine, dim=-1).view(-1, num_labels)  # (bs*seq_len, vocab_size)
                uniform_dist = torch.full_like(soft_outputs, 1.0 / num_labels).to(preds_fine.device)
                max_indices = torch.argmax(uniform_dist , dim=1)  # (bs*seq_len, vocab_size)
                kl_div = F.kl_div((soft_outputs + 1e-7).log(), uniform_dist, reduction='batchmean') # (bs*(seq_len - 1))
                loss=kl_div+ loss_coarse#+KL_loss(preds_coarse, origin_coarse) * args.KL_c + KL_loss(logits_fine_masked, origin_fine_masked) * args.KL_f
            else:
                num_labels_fine = preds_fine.shape[-1]
                soft_outputs_fine = F.softmax(preds_fine, dim=-1).view(-1, num_labels_fine)  # (bs*seq_len, vocab_size)
                uniform_dist_fine = torch.full_like(soft_outputs_fine, 1.0 / num_labels_fine).to(preds_fine.device)  # (bs*seq_len, vocab_size)
                kl_div_fine = F.kl_div((soft_outputs_fine + 1e-7).log(), uniform_dist_fine, reduction='batchmean').sum(-1)  # (bs*(seq_len - 1))
                num_labels_coarse = preds_coarse.shape[-1]
                soft_outputs_coarse = F.softmax(preds_coarse, dim=-1).view(-1, num_labels_coarse)  # (bs*seq_len, vocab_size)
                niform_dist_coarse = torch.full_like(soft_outputs_coarse, 1.0 / num_labels_coarse).to(preds_coarse.device)  # (bs*seq_len, vocab_size)
                kl_div_coarse = F.kl_div((soft_outputs_coarse + 1e-7).log(), uniform_dist_coarse, reduction='batchmean').sum(-1)  # (bs*(seq_len - 1))
                loss= kl_div_fine+loss_fine
        elif args.unlearning_method=='neg':
            if args.unlearning_goal=='fine':
                loss=loss_fine-0.05*loss_coarse
            else:
                loss=loss_coarse+loss_fine
        elif args.unlearning_method=='SALUN':
            if args.unlearning_goal=='fine':
                loss_random=F.cross_entropy(preds_fine, fine_labels)
                loss=loss_random
            else:
                loss_random=F.cross_entropy(preds_coarse, coarse_labels)
                loss=loss_random
        elif args.unlearning_method=='Relabeling':
            if args.unlearning_goal=='fine':
                loss_random=F.cross_entropy(preds_fine, fine_labels)
                loss=loss_random
            else:
                loss_random=F.cross_entropy(preds_coarse, coarse_labels)
                loss=loss_random
        else:
           print("Please introduce new methids")

        
        acc_coarse = count_acc(preds_coarse, coarse_labels)
        acc_fine = count_acc(preds_fine, fine_labels)
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            f'Unlearning Process, epo {epoch}, iter {iterations}, lrc={lrc:.6f}, total loss={loss.item():.4f}, acc_coarse={acc_coarse:.4f}, acc_fine={acc_fine:.4f}'
        )

        tl.add(loss.item())
        ta_coarse.add(acc_coarse)
        ta_fine.add(acc_fine)
        optimizer.zero_grad()
        loss.backward()
        if SALUN_mask:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= SALUN_mask[name]
        optimizer.step()
        
        # Log training stats to wandb
        wandb.log({
            "forget_train/it_coarse_loss": loss_coarse.item(),
            "forget_train/it_fine_loss": loss_fine.item(),
            "forget_train/it_train_loss": loss.item(),
            "forget_train/it_coarse_accuracy": acc_coarse,
            "forget_train/it_fine_accuracy": acc_fine,
            "forget_learning_rate": lrc,
            "iteration": iterations
        })

        # Test and log every log_interval iterations
        if iterations % log_interval == 0 and testloader is not None:
            model.eval()
            test_coarse_loss, test_fine_loss, test_coarse, test_fine = test(model, testloader, epoch, args)
            
            # Log testing stats to wandb
            wandb.log({
                "forget_test/it_coarse_loss": test_coarse_loss,
                "forget_test/it_fine_loss": test_fine_loss,
                "forget_test/it_coarse_accuracy": test_coarse,
                "forget_test/it_fine_accuracy": test_fine,
                "iteration": iterations,
            })
            model.train()

    # Return aggregated metrics for the epoch
    tl = tl.item()
    ta_fine = ta_fine.item()
    ta_coarse = ta_coarse.item()
    return tl, ta_coarse, ta_fine, iterations



def test(model, testloader, epoch, args):
    model = model.eval()
    vl_fine = Averager()
    vl_coarse = Averager()
    va_fine = Averager()
    va_coarse = Averager()

    fine_class_image_count = {cls_idx: 0 for cls_idx in range(len(model.fine_classnames))}
    coarse_class_image_count = {cls_idx: 0 for cls_idx in range(len(model.coarse_classnames))}

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        preds_list_coarse = []
        preds_list_fine = []
        labels_list_coarse = []
        labels_list_fine = []

        for i, batch in enumerate(tqdm_gen, 1):
            data, labels_coarse, labels_fine, path = [_ for _ in batch]
            data = data.cuda()
            labels_coarse = labels_coarse.cuda()
            labels_fine = labels_fine.cuda()

            for fine_label in labels_fine:
                fine_class_image_count[fine_label.item()] += 1
            for coarse_label in labels_coarse:
                coarse_class_image_count[coarse_label.item()] += 1

            preds_coarse, preds_fine, features = model(data, training=False)
            predicts_coarse = torch.max(preds_coarse, dim=1)[1]
            predicts_fine = torch.max(preds_fine, dim=1)[1]
            preds_list_coarse.append(predicts_coarse)
            preds_list_fine.append(predicts_fine)
            labels_list_coarse.append(labels_coarse)
            labels_list_fine.append(labels_fine)

            loss_fine = F.cross_entropy(preds_fine, labels_fine)
            loss_coarse = F.cross_entropy(preds_coarse, labels_coarse)
            acc_fine = count_acc(preds_fine, labels_fine)
            acc_coarse = count_acc(preds_coarse, labels_coarse)

            vl_fine.add(loss_fine.item())
            vl_coarse.add(loss_coarse.item())
            va_fine.add(acc_fine)
            va_coarse.add(acc_coarse)

        preds_list_coarse = torch.cat(preds_list_coarse, dim=0)
        labels_list_coarse = torch.cat(labels_list_coarse, dim=0)
        preds_list_fine = torch.cat(preds_list_fine, dim=0)
        labels_list_fine = torch.cat(labels_list_fine, dim=0)

        classes_fine = torch.unique(labels_list_fine)
        accuracy_per_class = {}
        coarse_accuracy_per_fine = {}

        for cls_fine in classes_fine:
            fine_indices = torch.where(labels_list_fine == cls_fine)
            fine_predictions = preds_list_fine[fine_indices]
            fine_labels = labels_list_fine[fine_indices]
            coarse_labels = labels_list_coarse[fine_indices]
            coarse_predictions = preds_list_coarse[fine_indices]

            correct_fine = torch.sum(fine_predictions == fine_labels).item()
            total_fine = len(fine_labels)
            fine_accuracy = correct_fine / total_fine
            accuracy_per_class[cls_fine.item()] = fine_accuracy

            correct_coarse = torch.sum(coarse_predictions == coarse_labels).item()
            total_coarse = len(coarse_labels)
            coarse_accuracy = correct_coarse / total_coarse
            coarse_accuracy_per_fine[cls_fine.item()] = coarse_accuracy

        for cls_fine, accuracy in accuracy_per_class.items():
            coarse_acc = coarse_accuracy_per_fine[cls_fine]
            fine_classname = model.fine_classnames[cls_fine]
            coarse_classname = model.coarse_classnames[labels_list_coarse[torch.where(labels_list_fine == cls_fine)[0][0]].item()]
            print(f"Fine Class {fine_classname}: Fine Accuracy = {accuracy:.3f}, Coarse Accuracy = {coarse_acc:.3f}, Coarse Class = {coarse_classname}")
            print(f"  Images in Fine Class: {fine_class_image_count[cls_fine]}")
            print(f"  Images in Coarse Class: {coarse_class_image_count[labels_list_coarse[torch.where(labels_list_fine == cls_fine)[0][0]].item()]}")

    vl_fine = vl_fine.item()
    vl_coarse = vl_coarse.item()
    va_coarse = va_coarse.item()
    va_fine = va_fine.item()

    print('epo {}, test,loss_coarse={:.4f} loss_fine={:.4f} acc_coarse={:.4f} acc_fine={:.4f}'.format(epoch, vl_coarse, vl_fine, va_coarse, va_fine))
    return vl_coarse, vl_fine, va_coarse, va_fine


