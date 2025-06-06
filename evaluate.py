import os
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
from dataloader.Compcars.car_dict import*
from dataloader.Subsetdog.subdog_dict import*
from collections import defaultdict

"""
Please specify the dataset root
"""
def get_evaluated_dataloader(args,test_transform):
    if args.evaluate_dataset=='imagenet':
        trainset=torchvision.datasets.ImageNet('/projectnb/ivc-ml/yuwentan/dataset/imagenet',split='train',transform=test_transform)
        testset=torchvision.datasets.ImageNet('/projectnb/ivc-ml/yuwentan/dataset/imagenet',split='val',transform=test_transform)
    if args.evaluate_dataset=='cifar100':
        trainset=torchvision.datasets.CIFAR100('/projectnb/ivc-ml/yuwentan/dataset/cifar100',train=True)
        testset=torchvision.datasets.CIFAR100('/projectnb/ivc-ml/yuwentan/dataset/cifar100',train=False,transform=test_transform)
    if args.evaluate_dataset=='Flower102':
        trainset=torchvision.datasets.Flowers102('/projectnb/ivc-ml/yuwentan/dataset/flower102',split='train',download=False)
        testset=torchvision.datasets.Flowers102('/projectnb/ivc-ml/yuwentan/dataset/flower102',split='test',download=False,transform=test_transform)
    if args.evaluate_dataset=='Stanford_Cars':
        trainset=torchvision.datasets.StanfordCars('/projectnb/ivc-ml/yuwentan/dataset/',split='train',transform=test_transform)
        testset=torchvision.datasets.StanfordCars('/projectnb/ivc-ml/yuwentan/dataset/',split='test',transform=test_transform)
    if args.evaluate_dataset=='caltech101':
        testset=Caltech101('/projectnb/ivc-ml/yuwentan/dataset/', target_type = 'category',transform = test_transform)
    if args.evaluate_dataset=='OxfordPet':
        trainset = torchvision.datasets.OxfordIIITPet(root='/projectnb/ivc-ml/yuwentan/dataset/OxfordIIITPet/', download=True, split='trainval', transform=test_transform)
        testset = torchvision.datasets.OxfordIIITPet(root='/projectnb/ivc-ml/yuwentan/dataset/OxfordIIITPet/', download=True, split='test', transform=test_transform)
    if args.evaluate_dataset=='Food101':
        trainset=torchvision.datasets.Food101(root='/projectnb/ivc-ml/yuwentan/dataset/Food101/', split = 'train', transform = test_transform)
        testset=torchvision.datasets.Food101(root='/projectnb/ivc-ml/yuwentan/dataset/Food101/', split = 'test', transform = test_transform)
    if args.evaluate_dataset=='SUN':
        testset=torchvision.datasets.SUN397(root='/projectnb/ivc-ml/yuwentan/dataset/SUN397/', transform = test_transform)
    return testset
    
def get_command_line_parser():
    parser = argparse.ArgumentParser()
    # about dataset and network
    parser.add_argument('-evaluate_dataset', type=str, default='Stanford_Cars', choices=['imagenet','caltech101','cifar100','Stanford_Cars','SUN397','OxfordPet','Flower102','Food101'])                  
    parser.add_argument('-template', type=str, default='Stanford_Cars',  choices=['imagenet','caltech101','cifar100','Stanford_Cars','SUN397','OxfordPet','Flower102','Food101'])
    parser.add_argument('-unlearning_method', type=str, default='GA', choices=['GDiff','GA','Relabeling','KL','SALUN','ME','neg','NPO','Hinge'])
    #Just for stanford_Cars
    parser.add_argument('-unlearn', action='store_true', help='Flag to activate unlearning')
    parser.add_argument('-unlearn_fine_classes',type=str,default=['Acura MDX','Lexus RX','Jaguar XK','MINI CABRIO','Audi A7','Audi A5 coupe','Cadillac SRX','Corvette','Mustang'],#['German short-haired pointer', 'Boston terrier', 'West Highland white terrier', 'Labrador retriever', 'golden retriever',  'German shepherd dog', 'keeshond',   'Samoyed', 'Pomeranian','Border terrier'],#['Irish setter','Gordon setter','basset hound','Airedale terrier','Shih-Tzu','miniature pinscher','Alaskan malamute','flat-coated retriever','Chesapeake Bay retriever','Sealyham terrier'],#['German short-haired pointer', 'Boston terrier', 'West Highland white terrier', 'Labrador retriever', 'golden retriever',  'German shepherd dog', 'keeshond',   'Samoyed', 'Pomeranian','Border terrier'],
    nargs='+', help='List of fine classes to unlearn.')
    parser.add_argument('-unlearn_coarse_classes',type=str,default=['Audi'],nargs='+', help='List of coarse classes to unlearn.')
    parser.add_argument('-unlearning_goal', type=str, default='fine', choices=['coarse','fine','hybrid'])
    parser.add_argument('-unlearn_dataset', type=str, default='Compcars',  choices=['Compcars', 'Tara','Subsetdog'])#['Audi A4L', 'Audi A7',  'Audi A8L', 'Audi S1', 'Audi R8', 'Audi A5 convertible','Audi TT RS'],#
    # about pre-training
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-test_batch_size', type=int, default=128)
    #/projectnb/ivc-ml/yuwentan/Unlearning/checkpoint/Compcars/Unlearning/Relabeling/Epo_6-Lr_0.00005-Step_40-Gam_0.10-Bs_32-Mom_0.90
    parser.add_argument('-model_dir', type=str, default='/projectnb/ivc-ml/yuwentan/Unlearning/checkpoint/Compcars/Unlearning/Hinge/fine/Cosine-Epo_6-Lr_0.00000010/difficult_KLc_20.0_KLf=20.0_Margin_2.0_beta_0.0.pth', help='loading model parameter from a specific dir')
    parser.add_argument('-backbonename', type=str, default='ViT-L/14', choices=['ViT-L/14','ViT-B/16'])
    # about training
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-seed', type=int, default=1)

    return parser

if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    # Define transformations and load data...
    test_transform =  transforms.Compose([
                transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])
    testset = get_evaluated_dataloader(args, test_transform)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True) 
    if args.evaluate_dataset=='imagenet':
        classnames=[t[0] for t in testset.classes]
        coarse_classnames=classnames
        fine_classnames=classnames
    elif args.evaluate_dataset=='Flower102':
        coarse_classnames=flower102_classnames
        fine_classnames=flower102_classnames
    elif args.evaluate_dataset=='caltech101':
        coarse_classnames=Caltech101_classnames
        fine_classnames=Caltech101_classnames
    elif args.evaluate_dataset=='Stanford_Cars':
        coarse_classnames= ['AM General', 'Acura', 'Aston Martin', 'Audi', 'BMW', 'Bentley', 'Bugatti', 'Buick', 'Cadillac', 'Chevrolet', 'Chrysler', 'Daewoo', 'Dodge', 'Eagle', 'FIAT', 'Ferrari', 'Fisker', 'Ford', 'GMC', 'Geo', 'HUMMER', 'Honda', 'Hyundai', 'Infiniti', 'Isuzu', 'Jaguar', 'Jeep', 'Lamborghini', 'Land Rover', 'Lincoln', 'MINI', 'Maybach','Mazda', 'McLaren', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Plymouth', 'Porsche', 'Ram', 'Rolls-Royce', 'Scion', 'Spyker', 'Suzuki', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo', 'smart']
        fine_classnames=testset.classes
    elif args.evaluate_dataset=='OxfordPet':
        coarse_classnames= ['domestic cats','hound','spaniel','setter','terrier','toy dog','bull dog','spitz','pinscher','pointer','other dogs']
        fine_classnames=testset.classes
        fine_classnames=['German short-haired pointer' if name == 'German Shorthaired' else name for name in fine_classnames]
    else:
        coarse_classnames=testset.classes 
        fine_classnames=testset.classes
    
    # Setup models...
    Unlearn_model = UnlearningCLIP(coarse_classnames, fine_classnames, args).cuda().eval()
    Origin_model = ZeroshotCLIP(coarse_classnames, fine_classnames, args).cuda().eval()
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
    if (args.evaluate_dataset=='Stanford_Cars' and args.unlearn_dataset=='Compcars') or (args.evaluate_dataset=='OxfordPet' and args.unlearn_dataset=='Subsetdog') :
        with torch.no_grad():
            preds_list_coarse = []
            labels_list_coarse = []
            preds_list_fine = []
            labels_list_fine = []
            if args.evaluate_dataset == "Stanford_Cars":
                active_dict = Stanford_Cars_car_dict
            elif args.evaluate_dataset == "OxfordPet":
                active_dict = OxfordPet_dict
            else:
                active_dict = {}  # or some default dictionary if needed
            for batch_idx, (data, fine_labels) in enumerate(testloader):
                data=data.cuda()
                fine_copy_labels = fine_labels.cpu().numpy()
                coarse_labels = []
                for fine_label in fine_copy_labels:
                    fine_name = fine_classnames[fine_label]
                    for coarse_name, fine_list in active_dict.items():
                        if fine_name in fine_list:
                            coarse_labels.append(coarse_classnames.index(coarse_name))
                            break
                coarse_labels=torch.tensor(coarse_labels)
                fine_labels = fine_labels.cuda()
                coarse_labels = coarse_labels.cuda()
                preds_coarse, preds_fine,features = Unlearn_model(data,training=False)
                predicts_fine = torch.max(preds_fine, dim=1)[1]
                preds_list_fine.append(predicts_fine)
                labels_list_fine.append(fine_labels)
                predicts_coarse = torch.max(preds_coarse, dim=1)[1]
                preds_list_coarse.append(predicts_coarse)
                labels_list_coarse.append(coarse_labels)
            preds_list_fine = torch.cat(preds_list_fine, dim=0)
            labels_list_fine = torch.cat(labels_list_fine, dim=0)
            preds_list_coarse = torch.cat(preds_list_coarse, dim=0)
            labels_list_coarse = torch.cat(labels_list_coarse, dim=0)
        classes = torch.unique(labels_list_fine)
        correct_coarse = torch.sum(preds_list_coarse == labels_list_coarse).item()  
        total_coarse = len(labels_list_coarse)  
        overall_accuracy_coarse = correct_coarse / total_coarse 
        print(f"Coarse Overall Accuracy: {overall_accuracy_coarse * 100:.2f}%")
        correct_fine = torch.sum(preds_list_fine == labels_list_fine).item()  
        total_fine = len(labels_list_fine)  
        overall_accuracy_fine = correct_fine / total_fine 
        print(f"Fine Overall Accuracy: {overall_accuracy_fine * 100:.2f}%")
        if args.unlearning_goal == 'coarse':
            class_indices = [i for i, class_name in enumerate(fine_classnames) if any(brand.lower() in class_name.lower() for brand in args.unlearn_coarse_classes)]
        if args.unlearning_goal == 'fine':
            class_indices = [i for i, class_name in enumerate(fine_classnames) if any(brand.lower() in class_name.lower() for brand in args.unlearn_fine_classes)]
        else:
            class_coarse_indices = [i for i, class_name in enumerate(fine_classnames) if any(brand.lower() in class_name.lower() for brand in args.unlearn_coarse_classes)]
            class_fine_indices = [i for i, class_name in enumerate(fine_classnames) if any(brand.lower() in class_name.lower() for brand in args.unlearn_fine_classes)]
            class_indices = class_coarse_indices + class_fine_indices
        total_unlearn_fine=0
        correct_unlearn_fine=0
        total_unlearn_coarse=0
        correct_unlearn_coarse=0
        total_retain_fine=0
        correct_retain_fine=0
        total_retain_coarse=0
        for i in range(len(labels_list_fine)):
            if labels_list_fine[i].item() in class_indices:  
                total_unlearn_coarse += 1
                total_unlearn_fine += 1
                if preds_list_coarse[i].item() == labels_list_coarse[i].item():  
                    correct_unlearn_coarse += 1
                if preds_list_fine[i].item() == labels_list_fine[i].item(): 
                    correct_unlearn_fine += 1    
            else:  
                total_retain_coarse += 1
                total_retain_fine += 1
                if preds_list_coarse[i].item() == labels_list_coarse[i].item(): 
                    correct_retain_coarse += 1
                if preds_list_fine[i].item() == labels_list_fine[i].item(): 
                    correct_retain_fine += 1
        unlearn_accuracy_coarse = correct_unlearn_coarse / total_unlearn_coarse if total_unlearn_coarse > 0 else 0
        unlearn_accuracy_fine = correct_unlearn_fine / total_unlearn_fine if total_unlearn_fine > 0 else 0
        others_accuracy_coarse = correct_retain_coarse / total_retain_coarse if total_retain_coarse> 0 else 0
        others_accuracy_fine = correct_retain_fine / total_retain_fine if total_retain_fine> 0 else 0
        print(f"Unlearn coarse accuracy: {unlearn_accuracy_coarse * 100:.2f}%")
        print(f"Unlearn fine accuracy: {unlearn_accuracy_fine * 100:.2f}%")
        print(f"Retain coarse accuracy : {others_accuracy_coarse * 100:.2f}%")
        print(f"Retain fine accuracy : {others_accuracy_fine * 100:.2f}%")
        class_correct_fine = defaultdict(int)
        class_total_fine = defaultdict(int)
        class_correct_coarse = defaultdict(int)
        class_total_coarse = defaultdict(int)

        for i in range(len(labels_list_fine)):
            fine_label_id = labels_list_fine[i].item()
            coarse_label_id = labels_list_coarse[i].item()

            if fine_label_id in class_indices:
                class_total_fine[fine_label_id] += 1
                if preds_list_fine[i] == labels_list_fine[i]:
                    class_correct_fine[fine_label_id] += 1

            if coarse_label_id in class_indices:
                class_total_coarse[coarse_label_id] += 1
                if preds_list_coarse[i] == labels_list_coarse[i]:
                    class_correct_coarse[coarse_label_id] += 1

        print("Class-wise Unlearn Accuracy (Fine):")
        for class_id, total in class_total_fine.items():
            accuracy = class_correct_fine[class_id] / total * 100
            print(f"Class {fine_classnames[class_id]}: {accuracy:.2f}%")
   
    else:
        with torch.no_grad():
            preds_list = []
            labels_list = []
            paths=[]
            for batch_idx, (data, labels) in enumerate(testloader):
                data=data.cuda()
                labels = labels.cuda()
                preds_coarse, preds_fine,features = Unlearn_model(data,training=False)
                #preds_coarse, preds_fine,features = Origin_model(data)
                predicts = torch.max(preds_fine, dim=1)[1]
                preds_list.append(predicts)
                labels_list.append(labels)
            preds_list = torch.cat(preds_list, dim=0)
            labels_list = torch.cat(labels_list, dim=0)
        classes = torch.unique(labels_list)
        correct = torch.sum(preds_list == labels_list).item()  
        total = len(labels_list)  
        overall_accuracy = correct / total 
        print(f"Unlearn_Overall Accuracy: {overall_accuracy * 100:.2f}%")  




    