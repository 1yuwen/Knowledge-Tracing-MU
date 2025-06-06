import sys
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io as sio
from collections import defaultdict
import os
import random
import json
import numpy as np
from torch.utils.data import DataLoader,Subset
import torchvision.datasets


class Subsetdog_Unlearn(Dataset):
    def __init__(self, root='/projectnb/ivc-ml/yuwentan/dataset/Compcars/', train=True,unlearn=False,goal='fine',unlearn_classes=[]):
        self.root = root
        self.train = train  # training set or test set
        if self.train:
            self.transform = transforms.Compose([
                transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])
            self._pre_operate(self.root,self.train)
            if  unlearn:
                if not unlearn_classes:
                    print("unlearn_classes is empty. Please provide a list of fine classes for unlearning.")
                    exit()
                else:
                    self.data, self.fine_targets,self.coarse_targets,self.index= self.SelectUnlearnClasses(self.data, self.fine_targets, self.coarse_targets,self.fine_texts,self.coarse_texts,goal,unlearn_classes)
            else:
                self.data, self.fine_targets, self.coarse_targets,self.index = self.SelectRetainClasses(self.data,self.fine_targets, self.coarse_targets, self.fine_texts,self.coarse_texts,goal,unlearn_classes)
            sampled_indices = self.sample_per_class()
            self.data = [self.data[i] for i in sampled_indices]
            self.fine_targets = [self.fine_targets[i] for i in sampled_indices]
            self.coarse_targets = [self.coarse_targets[i] for i in sampled_indices]
            self.index = [self.index[i] for i in sampled_indices]
        else:
            self.transform = transforms.Compose([
                transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ])
            self._pre_operate(self.root,self.train)
            if unlearn:
                if not unlearn_classes:
                    print("unlearn_fine_classes is empty. Please provide a list of fine classes for unlearning.")
                    exit()
                else:
                    self.data, self.fine_targets, self.coarse_targets,self.index = self.SelectUnlearnClasses(self.data,  self.fine_targets, self.coarse_targets,self.fine_texts,self.coarse_texts,goal,unlearn_classes)
            else:
                self.data, self.fine_targets, self.coarse_targets,self.index = self.SelectRetainClasses(self.data, self.fine_targets,self.coarse_targets, self.fine_texts,self.coarse_texts,goal,unlearn_classes)

    
    def sample_per_class(self):
        class_to_indices = {}
        for idx in range(len(self.fine_targets)):
            label = self.fine_targets[idx] 
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)

        sampled_indices = []
        for label, indices in class_to_indices.items():
            max_samples = min(len(indices), 150) 
            sampled_indices.extend(random.sample(indices, max_samples))

        return sampled_indices

    def SelectRetainClasses(self, data, fine_targets, coarse_targets, fine_texts,coarse_texts, goal,unlearn_classes):
        if goal=='fine':
            data_tmp = [data[i] for i in range(len(data)) if fine_texts[i] not in unlearn_classes]
            targets_fine = [fine_targets[i] for i in range(len(data)) if fine_texts[i] not in unlearn_classes]
            targets_coarse = [coarse_targets[i] for i in range(len(data)) if fine_texts[i] not in unlearn_classes]
            index=[1 for i in range(len(data)) if fine_texts[i] not in unlearn_classes]
        elif goal=='coarse':
            data_tmp = [data[i] for i in range(len(data)) if coarse_texts[i] not in unlearn_classes]
            targets_fine = [fine_targets[i] for i in range(len(data)) if coarse_texts[i] not in unlearn_classes]
            targets_coarse = [coarse_targets[i] for i in range(len(data)) if coarse_texts[i] not in unlearn_classes]
            index=[0 for i in range(len(data)) if coarse_texts[i] not in unlearn_classes]
        elif goal=='hybrid':
            data_tmp = [data[i] for i in range(len(data)) if fine_texts[i] not in unlearn_classes and coarse_texts[i] not in unlearn_classes]
            targets_fine = [fine_targets[i] for i in range(len(data)) if fine_texts[i] not in unlearn_classes and coarse_texts[i] not in unlearn_classes]
            targets_coarse = [coarse_targets[i] for i in range(len(data)) if fine_texts[i] not in unlearn_classes and coarse_texts[i] not in unlearn_classes]
            index=[2 for i in range(len(data)) if fine_texts[i] not in unlearn_classes and coarse_texts[i] not in unlearn_classes]
        return data_tmp, targets_fine, targets_coarse,index


    def SelectUnlearnClasses(self, data, fine_targets,coarse_targets, fine_texts,coarse_texts,goal, unlearn_classes):
        data_tmp = []
        targets_fine = []
        targets_coarse=[]
        index=[]
        #coarse '0',fine,'1'
        for i in unlearn_classes:
            if goal=='fine':
                ind_cl = [j for j, x in enumerate(fine_texts) if x == i]
                if len(ind_cl) == 0:
                    print(i)
                for j in ind_cl:
                    data_tmp.append(data[j])
                    targets_fine.append(fine_targets[j])
                    targets_coarse.append(coarse_targets[j])
                    index.append(1)
            else:
                ind_cl = [j for j, x in enumerate(coarse_texts) if x == i]
                if len(ind_cl) == 0:
                    print(i)
                for j in ind_cl:
                    data_tmp.append(data[j])
                    targets_fine.append(fine_targets[j])
                    targets_coarse.append(coarse_targets[j])
                    index.append(0)
        return data_tmp, targets_fine,targets_coarse,index
        


    def _pre_operate(self, root,train):
        image_file=[]
        if train:
            json_file = os.path.join(root,'train.jsonl')
        else:
            json_file = os.path.join(root,'test.jsonl')
        with open(json_file, "r") as f:
            for line in f:
                image_file.append(json.loads(line))
        with open('/projectnb/ivc-ml/yuwentan/Unlearning/data/Subsetdog/coarse_label_subset.txt', 'r') as file:
                coarse_names_set = file.read().splitlines()
        with open('/projectnb/ivc-ml/yuwentan/Unlearning/data/Subsetdog/fine_label_subset.txt', 'r') as file:
                fine_names_set = file.read().splitlines()
        coarse_dict = {value: index for index, value in enumerate(list(coarse_names_set))}
        fine_dict = {value: index for index, value in enumerate(list(fine_names_set))}
        self.data = []
        self.coarse_targets = []
        self.fine_targets = []
        self.coarse_texts = []
        self.fine_texts = []
        self.coarse_targets = []
        self.fine_targets = []
        for k in image_file:
            image_path = k["img_path"]
            fine_text=k["fine_label"]
            coarse_text=k["coarse_label"]
            coarse_label = coarse_dict[coarse_text]
            fine_label =  fine_dict[fine_text]
            self.data.append(image_path)
            self.coarse_targets.append(coarse_label)
            self.fine_targets.append(fine_label)
            self.coarse_texts.append(coarse_text)
            self.fine_texts.append(fine_text)
        print('finished')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, fine_targets,coarse_targets,index= self.data[i], self.fine_targets[i],self.coarse_targets[i],self.index[i]
        image = Image.open(path).convert('RGB')
        classify_image = self.transform(image)
        total_image = classify_image
        return total_image, coarse_targets,fine_targets, index#,text


# if __name__ == "__main__":
#     unlearn_train_dataset=Subsetdog_Unlearn(root='/projectnb/ivc-ml/yuwentan/Unlearning/data/Subsetdog/', train=False,unlearn=True,goal='coarse',unlearn_classes=['terrier'])
#     print(1)