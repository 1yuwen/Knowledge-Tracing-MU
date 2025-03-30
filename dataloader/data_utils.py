import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import ConcatDataset

def set_up_datasets(args):
    if args.unlearn_dataset == 'Compcars':
        import dataloader.Compcars.Compcars as unlearn_Dataset
    if args.unlearn_dataset == 'Subsetdog':
        import dataloader.Subsetdog.Subsetdog as unlearn_Dataset
    args.unlearn_Dataset=unlearn_Dataset
    return args

def get_unlearn_dataloader(args):
    if args.unlearn_dataset == 'Compcars':
        unlearn_fine_classes = args.unlearn_fine_classes
        unlearn_coarse_classes= args.unlearn_coarse_classes
        unlearning_goal=args.unlearning_goal
        if args.unlearning_goal=='fine':
            unlearn_trainset = args.unlearn_Dataset.COMPARS_Unlearn( root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=True,unlearn=True,goal=unlearning_goal,unlearn_classes= unlearn_fine_classes,trans=True)
            unlearn_testset = args.unlearn_Dataset.COMPARS_Unlearn(root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=False,unlearn=True,goal=unlearning_goal,unlearn_classes= unlearn_fine_classes,trans=True)
        if args.unlearning_goal=='coarse':
            unlearn_trainset = args.unlearn_Dataset.COMPARS_Unlearn( root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=True,unlearn=True,goal=unlearning_goal,unlearn_classes= unlearn_coarse_classes,trans=True)
            unlearn_testset = args.unlearn_Dataset.COMPARS_Unlearn(root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=False,unlearn=True,goal=unlearning_goal,unlearn_classes= unlearn_coarse_classes,trans=True)
        if args.unlearning_goal=='hybrid':
            unlearn_coarse_trainset = args.unlearn_Dataset.COMPARS_Unlearn( root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=True,unlearn=True,goal='coarse',unlearn_classes= unlearn_coarse_classes,trans=True)
            unlearn_coarse_testset = args.unlearn_Dataset.COMPARS_Unlearn(root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=False,unlearn=True,goal='coarse',unlearn_classes= unlearn_coarse_classes,trans=True)   
            unlearn_fine_trainset = args.unlearn_Dataset.COMPARS_Unlearn( root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=True,unlearn=True,goal='fine',unlearn_classes= unlearn_coarse_classes,trans=True)
            unlearn_fine_testset = args.unlearn_Dataset.COMPARS_Unlearn(root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=False,unlearn=True,goal='fine',unlearn_classes= unlearn_coarse_classes,trans=True)     
            unlearn_trainset= ConcatDataset([unlearn_coarse_trainset,unlearn_fine_trainset])  
            unlearn_testset= ConcatDataset([unlearn_coarse_testset,unlearn_fine_testset])
    if args.unlearn_dataset == 'Subsetdog':
        unlearn_fine_classes = args.unlearn_fine_classes
        unlearn_coarse_classes= args.unlearn_coarse_classes
        unlearning_goal=args.unlearning_goal
        if args.unlearning_goal=='fine':
            unlearn_trainset = args.unlearn_Dataset.Subsetdog_Unlearn( root='/projectnb/ivc-ml/yuwentan/dataset/imagenet/', train=True,unlearn=True,goal=unlearning_goal,unlearn_classes= unlearn_fine_classes)
            unlearn_testset = args.unlearn_Dataset.Subsetdog_Unlearn(root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=False,unlearn=True,goal=unlearning_goal,unlearn_classes= unlearn_fine_classes)
        if args.unlearning_goal=='coarse':
            unlearn_trainset = args.unlearn_Dataset.Subsetdog_Unlearn( root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=True,unlearn=True,goal=unlearning_goal,unlearn_classes= unlearn_coarse_classes)
            unlearn_testset = args.unlearn_Dataset.Subsetdog_Unlearn(root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=False,unlearn=True,goal=unlearning_goal,unlearn_classes= unlearn_coarse_classes)
        if args.unlearning_goal=='hybrid':
            unlearn_coarse_trainset = args.unlearn_Dataset.Subsetdog_Unlearn( root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=True,unlearn=True,goal='coarse',unlearn_classes= unlearn_coarse_classes)
            unlearn_coarse_testset = args.unlearn_Dataset.Subsetdog_Unlearn(root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=False,unlearn=True,goal='coarse',unlearn_classes= unlearn_coarse_classes)   
            unlearn_fine_trainset = args.unlearn_Dataset.Subsetdog_Unlearn( root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=True,unlearn=True,goal='fine',unlearn_classes= unlearn_coarse_classes)
            unlearn_fine_testset = args.unlearn_Dataset.Subsetdog_Unlearn(root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=False,unlearn=True,goal='fine',unlearn_classes= unlearn_coarse_classes)     
            unlearn_trainset= ConcatDataset([unlearn_coarse_trainset,unlearn_fine_trainset])  
            unlearn_testset= ConcatDataset([unlearn_coarse_testset,unlearn_fine_testset])
    unlearn_trainloader = torch.utils.data.DataLoader(dataset=unlearn_trainset, batch_size=args.batch_size_base, shuffle=True,num_workers=8, pin_memory=True)
    unlearn_testloader = torch.utils.data.DataLoader( dataset=unlearn_testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return unlearn_trainloader, unlearn_testloader

def get_retain_dataloader(args):
    unlearn_fine_classes = args.unlearn_fine_classes
    unlearn_coarse_classes= args.unlearn_coarse_classes
    unlearning_goal=args.unlearning_goal
    if args.unlearn_dataset == 'Compcars':
        if args.unlearning_goal=='fine':
            retain_trainset = args.unlearn_Dataset.COMPARS_Unlearn( root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=True,unlearn=False,goal=unlearning_goal,unlearn_classes= unlearn_fine_classes,trans=True)
            retain_testset = args.unlearn_Dataset.COMPARS_Unlearn(root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=False,unlearn=False,goal=unlearning_goal,unlearn_classes= unlearn_fine_classes,trans=True)
        elif args.unlearning_goal=='coarse':
            retain_trainset = args.unlearn_Dataset.COMPARS_Unlearn( root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=True,unlearn=False,goal=unlearning_goal,unlearn_classes= unlearn_coarse_classes,trans=True)
            retain_testset = args.unlearn_Dataset.COMPARS_Unlearn(root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=False,unlearn=False,goal=unlearning_goal,unlearn_classes= unlearn_coarse_classes,trans=True)
        elif args.unlearning_goal=='hybrid':
            unlearn_classes=unlearn_fine_classes+unlearn_coarse_classes
            retain_trainset = args.unlearn_Dataset.COMPARS_Unlearn( root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=True,unlearn=False,goal=unlearning_goal,unlearn_classes= unlearn_classes,trans=True)
            retain_testset = args.unlearn_Dataset.COMPARS_Unlearn(root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=False,unlearn=False,goal=unlearning_goal,unlearn_classes= unlearn_classes,trans=True)   
    if args.unlearn_dataset == 'Subsetdog':
        if args.unlearning_goal=='fine':
            retain_trainset = args.unlearn_Dataset.Subsetdog_Unlearn( root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=True,unlearn=False,goal=unlearning_goal,unlearn_classes= unlearn_fine_classes)
            retain_testset = args.unlearn_Dataset.Subsetdog_Unlearn(root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=False,unlearn=False,goal=unlearning_goal,unlearn_classes= unlearn_fine_classes)
        elif args.unlearning_goal=='coarse':
            retain_trainset = args.unlearn_Dataset.Subsetdog_Unlearn( root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=True,unlearn=False,goal=unlearning_goal,unlearn_classes= unlearn_coarse_classes)
            retain_testset = args.unlearn_Dataset.Subsetdog_Unlearn(root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=False,unlearn=False,goal=unlearning_goal,unlearn_classes= unlearn_coarse_classes)
        elif args.unlearning_goal=='hybrid':
            unlearn_classes=unlearn_fine_classes+unlearn_coarse_classes
            retain_trainset = args.unlearn_Dataset.Subsetdog_Unlearn( root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=True,unlearn=False,goal=unlearning_goal,unlearn_classes= unlearn_classes)
            retain_testset = args.unlearn_Dataset.Subsetdog_Unlearn(root='/projectnb/ivc-ml/dataset/Compcars_subset/', train=False,unlearn=False,goal=unlearning_goal,unlearn_classes= unlearn_classes)      
    retain_trainloader = torch.utils.data.DataLoader(dataset=retain_trainset, batch_size=args.batch_size_base, shuffle=True,num_workers=8, pin_memory=True)
    retain_testloader = torch.utils.data.DataLoader(dataset=retain_testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return retain_trainloader, retain_testloader