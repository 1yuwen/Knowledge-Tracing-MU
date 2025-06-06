from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io as sio
import os
import json
import numpy as np

class COMPARS_Unlearn(Dataset):
    def __init__(self, root='./data/Compcars', train=True,unlearn=False,goal='fine',unlearn_classes=[],trans=True):
        self.root = root
        self.train = train  
        if trans:
            if self.train:
                self.transform = transforms.Compose([
                    transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
                ])
            else:
                self.transform = transforms.Compose(
                    [   
                        transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
                    ])
        else:
               self.transform = None
        self._pre_operate(self.root, self.train)
        if unlearn:
            if not unlearn_classes:
                print("unlearn_classes is empty. Please provide a list of fine classes for unlearning.")
                exit()
            else:
                self.data, self.fine_targets, self.coarse_targets, self.index = self.SelectUnlearnClasses(self.data, self.fine_targets, self.coarse_targets, self.fine_texts, self.coarse_texts, goal, unlearn_classes)
        else:
            self.data, self.fine_targets, self.coarse_targets, self.index = self.SelectRetainClasses(self.data, self.fine_targets, self.coarse_targets, self.fine_texts, self.coarse_texts, goal, unlearn_classes)


    def SelectRetainClasses(self, data, fine_targets, coarse_targets, fine_texts,coarse_texts, goal,unlearn_classes):
        if goal=='fine':
            data_tmp = [data[i] for i in range(len(data)) if fine_texts[i] not in unlearn_classes]
            targets_fine = [fine_targets[i] for i in range(len(data)) if fine_texts[i] not in unlearn_classes]
            targets_coarse = [coarse_targets[i] for i in range(len(data)) if fine_texts[i] not in unlearn_classes]
            index=[1 for i in range(len(data)) if fine_texts[i] not in unlearn_classes]
        else:
            data_tmp = [data[i] for i in range(len(data)) if coarse_texts[i] not in unlearn_classes]
            targets_fine = [fine_targets[i] for i in range(len(data)) if coarse_texts[i] not in unlearn_classes]
            targets_coarse = [coarse_targets[i] for i in range(len(data)) if coarse_texts[i] not in unlearn_classes]
            index=[0 for i in range(len(data)) if coarse_texts[i] not in unlearn_classes]
        return data_tmp, targets_fine, targets_coarse,index


    def SelectUnlearnClasses(self, data, fine_targets,coarse_targets, fine_texts,coarse_texts,goal, unlearn_classes):
        data_tmp = []
        targets_fine = []
        targets_coarse=[]
        index=[]
        '''
        coarse:0 fine:1
        '''
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
        root='./data/Compcars'
        image_file=[]
        if train:
            json_file = os.path.join(root,'train.jsonl')
        else:
            json_file = os.path.join(root,'test.jsonl')
        with open(json_file, "r") as f:
            for line in f:
                image_file.append(json.loads(line))
        self.data = []
        self.coarse_texts = []
        self.fine_texts = []
        for k in image_file:
            image_path = k["img_path"]
            self.data.append(image_path)
            self.coarse_texts.append(k["coarse_label"])
            self.fine_texts.append(k["fine_label"])
        with open('./data/Compcars/coarse_label_subset.txt', 'r') as file:
            coarse_names_set = file.read().splitlines()
        with open('./data/Compcars/fine_label_subset.txt', 'r') as file:
            fine_names_set = file.read().splitlines()
        coarse_dict = {value: index for index, value in enumerate(list(coarse_names_set))}
        fine_dict = {value: index for index, value in enumerate(list(fine_names_set))}
        self.coarse_targets = []
        self.fine_targets = []
        for i in range(len(self.coarse_texts)):
            self.coarse_targets.append(coarse_dict[self.coarse_texts[i]])
            self.fine_targets.append(fine_dict[self.fine_texts[i]])
        print('finished')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, fine_targets,coarse_targets,index= self.data[i], self.fine_targets[i],self.coarse_targets[i],self.index[i]
        if self.transform:
            image = Image.open(path).convert('RGB')
            classify_image = self.transform(image)
        else:
            classify_image = Image.open(path)
        total_image = classify_image
        return total_image, coarse_targets,fine_targets, index

# if __name__ == "__main__":
#     unlearn_train_dataset=COMPARS_Unlearn(root='/projectnb/ivc-ml/yuwentan/dataset/Compcars/', train=True,unlearn=True,goal='fine',unlearn_classes=['BMW i3'])
#     print(1)