import os
import json
import argparse
import importlib
from utils import *
import os.path as osp
from copy import deepcopy
from utils import *
from model.Network import*
from dataloader.data_utils import *
from Unlearning.helper import*
from model.class_names import*
from dataloader.caltech101 import*
from dataloader.Compcars.Compcars import*
from dataloader.Subsetdog.Subsetdog import*
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from Unlearning.utils_helper import*
from sklearn.preprocessing import LabelEncoder



def get_evaluated_dataloader(args):
    if args.unlearn_dataset=='Compcars':
        retain_trainset = COMPARS_Unlearn( root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=True,unlearn=True,goal=args.unlearning_goal,unlearn_classes=args.unlearn_fine_classes,trans=True)
        retain_testset = COMPARS_Unlearn(root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=False,unlearn=args.unlearn,goal=args.unlearning_goal,unlearn_classes= args.unlearn_fine_classes,trans=True)
    if args.unlearn_dataset=='Subsetdog':
        retain_trainset = Subsetdog_Unlearn( root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=True,unlearn=True,goal=args.unlearning_goal,unlearn_classes=args.unlearn_fine_classes)
        retain_testset = Subsetdog_Unlearn(root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=False,unlearn=args.unlearn,goal=args.unlearning_goal,unlearn_classes= args.unlearn_fine_classes)
    return  retain_testset
    
def get_command_line_parser():
    parser = argparse.ArgumentParser()
    # setting of the dataset and the unlearn task
    parser.add_argument('-unlearning_goal', type=str, default='fine', choices=['fine','coarse','hybrid'])
    parser.add_argument('-unlearn_dataset', type=str, default='Subsetdog',  choices=['Compcars', 'Subsetdog'])
    parser.add_argument('-template', type=str, default='Subsetdog_test',  choices=['Subsetdog_test','Compcars_test'])
    parser.add_argument('-unlearn_fine_classes',type=str,default=['German short-haired pointer', 'Boston terrier', 'West Highland white terrier', 'Labrador retriever', 'golden retriever',  'German shepherd dog', 'keeshond',   'Samoyed', 'Pomeranian','Border terrier'],
    nargs='+', help='List of fine classes to unlearn.')
    parser.add_argument('-unlearn', action='store_true', help='Flag to activate unlearning')
    parser.add_argument('-unlearn_coarse_classes',type=str,default= ['Audi'],nargs='+', help='List of classes to unlearn.')             
    parser.add_argument('-unlearning_method', type=str, default='GDiff', choices=['GDiff','GA','Relabeling','KL','SALUN','ME','neg','NPO','Hinge'])
    # about pre-training
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-test_batch_size', type=int, default=128)
    parser.add_argument('-model_dir', type=str, default='/projectnb/ivc-ml/yuwentan/Unlearning/checkpoint/Subsetdog/Unlearning/SALUN/fine/Cosine-Epo_8-Lr_0.00000010/difficult_fine_min_acc.pth', help='loading model parameter from a specific dir')
    parser.add_argument('-backbonename', type=str, default='ViT-L/14', choices=['ViT-L/14','ViT-B/16'])
    # about training
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-seed', type=int, default=1)

    return parser


def calculate_class_accuracy(preds, labels, class_names):
    class_stats = {cls: {"correct": 0, "total": 0} for cls in class_names}
    for pred, label in zip(preds, labels):
        class_name = class_names[label.item()]  
        class_stats[class_name]["total"] += 1
        if pred == label:
            class_stats[class_name]["correct"] += 1

    class_accuracies = {
        cls: stat["correct"] / stat["total"] if stat["total"] > 0 else 0.0
        for cls, stat in class_stats.items()
    }
    return class_accuracies


def calculate_overall_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


'''
Compute accuracy of each class
'''

if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))

    # Load the test dataset
    testset = get_evaluated_dataloader(args)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Load class names based on the dataset
    if args.unlearn_dataset == 'Compcars':
        coarse_classnames = list(text_read('/projectnb/ivc-ml/yuwentan/Unlearning/dataloader/Compcars/coarse_label_subset.txt'))
        fine_classnames = list(text_read('/projectnb/ivc-ml/yuwentan/Unlearning/dataloader/Compcars/fine_label_subset.txt'))
    if args.unlearn_dataset == 'Subsetdog':
        coarse_classnames = list(text_read('/projectnb/ivc-ml/yuwentan/Unlearning/dataloader/Subsetdog/coarse_label_subset.txt'))
        fine_classnames = list(text_read('/projectnb/ivc-ml/yuwentan/Unlearning/dataloader/Subsetdog/fine_label_subset.txt'))

    # Load the model
    Unlearn_model = UnlearningCLIP(coarse_classnames, fine_classnames, args)
    Unlearn_model = Unlearn_model.cuda()
    Origin_model = ZeroshotCLIP(coarse_classnames, fine_classnames, args)
    Origin_model = Origin_model.cuda()

    if args.model_dir is not None:
        print('Loading init parameters from: %s' % args.model_dir)
        load_pretrained_weights(Unlearn_model, args.model_dir)
    else:
        print('Using the original pre-trained CLIP')

    if args.unlearning_method == 'neg':
        Unlearn_model_statedict = Unlearn_model.state_dict()
        Origin_model_statedict = Origin_model.state_dict()
        task_vector = TaskVector(Origin_model_statedict, Unlearn_model_statedict)
        new_dict = task_vector.apply_to(Origin_model_statedict, scaling_coef=1.5)
        Unlearn_model.load_state_dict(new_dict, strict=False)

    # Evaluate the model
    Unlearn_model = Unlearn_model.eval()
    Origin_model = Origin_model.eval()

    # Initialize dictionaries to store counts for accuracy calculations
    correct_count_coarse = {}
    total_count_coarse = {}
    correct_count_fine = {}
    total_count_fine = {}

    with torch.no_grad():
        preds_list_coarse = []
        preds_list_fine = []
        labels_list_coarse = []
        labels_list_fine = []
        confidences_coarse = []
        confidences_fine = []

        for batch_idx, (data, labels_coarse, labels_fine, _) in enumerate(testloader):
            data = data.cuda()
            labels_coarse = labels_coarse.cuda()
            labels_fine = labels_fine.cuda()

            # Model predictions
            preds_coarse, preds_fine, features = Unlearn_model(data, training=False)

            # Get probabilities (confidence scores)
            probs_coarse = F.softmax(preds_coarse, dim=1)
            probs_fine = F.softmax(preds_fine, dim=1)

            # Extract Top-1 confidence and predicted labels
            top1_coarse_confidence, predicts_coarse = torch.max(probs_coarse, dim=1)
            top1_fine_confidence, predicts_fine = torch.max(probs_fine, dim=1)

            # Append predictions and confidences
            preds_list_coarse.append(predicts_coarse)
            preds_list_fine.append(predicts_fine)
            labels_list_coarse.append(labels_coarse)
            labels_list_fine.append(labels_fine)
            confidences_coarse.append(top1_coarse_confidence)
            confidences_fine.append(top1_fine_confidence)

            # Update counts for coarse labels
            for label, pred in zip(labels_coarse, predicts_coarse):
                label = label.item()
                pred = pred.item()
                if label not in total_count_coarse:
                    total_count_coarse[label] = 0
                    correct_count_coarse[label] = 0
                total_count_coarse[label] += 1
                if pred == label:
                    correct_count_coarse[label] += 1

            # Update counts for fine labels
            for label, pred in zip(labels_fine, predicts_fine):
                label = label.item()
                pred = pred.item()
                if label not in total_count_fine:
                    total_count_fine[label] = 0
                    correct_count_fine[label] = 0
                total_count_fine[label] += 1
                if pred == label:
                    correct_count_fine[label] += 1

        # Combine all batches
        labels_list_coarse = torch.cat(labels_list_coarse, dim=0)
        labels_list_fine = torch.cat(labels_list_fine, dim=0)
        preds_list_coarse = torch.cat(preds_list_coarse, dim=0)
        preds_list_fine = torch.cat(preds_list_fine, dim=0)
        confidences_coarse = torch.cat(confidences_coarse, dim=0).cpu().numpy()
        confidences_fine = torch.cat(confidences_fine, dim=0).cpu().numpy()

        # Calculate overall accuracies
        correct_coarse = torch.sum(preds_list_coarse == labels_list_coarse).item()
        total_coarse = len(labels_list_coarse)
        overall_accuracy_coarse = correct_coarse / total_coarse
        print(f"Retain Coarse Accuracy: {overall_accuracy_coarse * 100:.3f}%")

        correct_fine = torch.sum(preds_list_fine == labels_list_fine).item()
        total_fine = len(labels_list_fine)
        overall_accuracy_fine = correct_fine / total_fine
        print(f"Retain Fine Accuracy: {overall_accuracy_fine * 100:.3f}%")

        # Calculate and print accuracy for each class in coarse categories
        print("Coarse Category Accuracies:")
        for label in total_count_coarse:
            accuracy = correct_count_coarse[label] / total_count_coarse[label] * 100
            print(f"Class {label}: Accuracy = {accuracy:.2f}%, Correct = {correct_count_coarse[label]}, Total = {total_count_coarse[label]}")

        # Calculate and print accuracy for each class in fine categories
        print("Fine Category Accuracies:")
        for label in total_count_fine:
            accuracy = correct_count_fine[label] / total_count_fine[label] * 100
            print(f"Class {label}: Accuracy = {accuracy:.2f}%, Correct = {correct_count_fine[label]}, Total = {total_count_fine[label]}")

