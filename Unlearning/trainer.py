import os.path as osp
import torch.nn as nn
from copy import deepcopy
from utils import *
from model.Network import*
from dataloader.data_utils import *
from .base import*
from .helper import*
from .utils_helper import*
from model.class_names import*
import wandb
                
class ULTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)
        if  args.unlearn_dataset == 'Compcars':
            coarse_classnames= list(text_read('./Unlearning/dataloader/Compcars/coarse_label_subset.txt'))
            fine_classnames = list(text_read('./Unlearning/dataloader/Compcars/fine_label_subset.txt'))
        if  args.unlearn_dataset == 'Subsetdog':
            coarse_classnames= list(text_read('./Unlearning/dataloader/Subsetdog/coarse_label_subset.txt'))
            fine_classnames = list(text_read('./Unlearning/dataloader/Subsetdog/fine_label_subset.txt'))
        self.model = UnlearningCLIP(coarse_classnames,fine_classnames,args)
        self.model = self.model.cuda()
        self.originmodel = ZeroshotCLIP(coarse_classnames, fine_classnames,args)
        self.originmodel  = self.originmodel.cuda()
        self.coarse_classnames=coarse_classnames
        self.fine_classnames=fine_classnames
        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
        else:
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):
        total_tuning_size = 0
        for name, param in self.originmodel.named_parameters():
           param.requires_grad_(False)
        for name, param in self.model.named_parameters():
            param.requires_grad_(True)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                total_tuning_size+=param.numel()
        print('Total Tuning Params:',total_tuning_size)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr,eps=1e-6,weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        return optimizer, scheduler

    def get_dataloader(self):
        unlearn_trainloader, unlearn_testloader = get_unlearn_dataloader(self.args)
        return unlearn_trainloader, unlearn_testloader
    
    def get_retain_dataloader(self):
        retain_trainloader, retain_testloader = get_retain_dataloader(self.args)
        return retain_trainloader, retain_testloader
    
    def get_relabeling_dataloader(self):
        unlearn_trainloader, unlearn_testloader = get_unlearn_dataloader(self.args)
        if self.args.unlearning_goal=='fine':
            fine_labels_names=[self.fine_classnames[idx] for idx in unlearn_trainloader.dataset.fine_targets]
            random_fine_labels=find_random_fine_models(fine_labels_names,self.fine_classnames)
            new_fine_labels=[self.fine_classnames.index(car) for car in random_fine_labels]
            unlearn_trainloader.dataset.fine_targets = new_fine_labels
        if self.args.unlearning_goal=='coarse':
            coarse_labels_names=[self.coarse_classnames[idx] for idx in unlearn_trainloader.dataset.coarse_targets]
            random_coarse_labels=find_random_coarse_models(coarse_labels_names)
            new_coarse_labels=[self.coarse_classnames.index(car) for car in random_coarse_labels]
            unlearn_trainloader.dataset.coarse_targets = new_coarse_labels
        return unlearn_trainloader, unlearn_testloader

    def train(self):
        args = self.args
        t_start_time = time.time()
        iterations=0
        log_interval=8
        project_name = f"Unlearning-project-difficult-unlearn_num-{args.unlearn_dataset}-{args.unlearning_goal}"
        if args.unlearning_method=='NPO':
            wandb_name = f"{args.unlearning_method}-lr={args.lr}-epochs={args.epochs}-KLc={args.KL_c}-KLf={args.KL_f}-beta={args.beta}"
        elif args.unlearning_method=='Hinge':
            wandb_name = f"{args.unlearning_method}-lr={args.lr}-epochs={args.epochs}-KLc={args.KL_c}-KLf={args.KL_f}-Hinge={args.margin_f}"
        elif args.unlearning_method=='KL':
             wandb_name = f"{args.unlearning_method}-lr={args.lr}-epochs={args.epochs}-KLc={args.KL_c}-KLf={args.KL_f}"
        else:
             wandb_name = f"{args.unlearning_method}-lr={args.lr}-epochs={args.epochs}"
        wandb.init(
            project=project_name,
            name=wandb_name
        )
        wandb.define_metric("epoch")
        wandb.define_metric("forget_train/epoch_train_loss", step_metric="epoch")
        wandb.define_metric("forget_test/epoch_coarse_acc", step_metric="epoch")
        wandb.define_metric("forget_train/epoch_fine_acc", step_metric="epoch")
        wandb.define_metric("forget_train/epoch_coarse_acc", step_metric="epoch")
        wandb.define_metric("forget_test/epoch_coarse_accuracy", step_metric="epoch")
        wandb.define_metric("forget_test/epoch_fine_accuracy", step_metric="epoch")      
        wandb.define_metric("iteration")
        wandb.define_metric("forget_train/it_coarse_loss", step_metric="iteration")
        wandb.define_metric("forget_train/it_fine_loss", step_metric="iteration")
        wandb.define_metric("forget_train/it_train_loss", step_metric="iteration")
        wandb.define_metric("forget_train/it_coarse_accuracy", step_metric="iteration")
        wandb.define_metric("forget_train/it_fine_accuracy", step_metric="iteration")
        wandb.define_metric("forget_test/it_coarse_loss", step_metric="iteration")
        wandb.define_metric("forget_test/it_fine_loss", step_metric="iteration")
        wandb.define_metric("forget_test/it_coarse_accuracy", step_metric="iteration")
        wandb.define_metric("forget_test/it_fine_accuracy", step_metric="iteration")

        if args.unlearning_method=='Relabeling'or args.unlearning_method == 'SALUN':
            unlearn_trainloader, unlearn_testloader= self.get_relabeling_dataloader()
        else:
            unlearn_trainloader, unlearn_testloader= self.get_dataloader()
        optimizer, scheduler = self.get_optimizer_base()
        for epoch in range(args.epochs):
            if args.unlearning_method=='SALUN':
                #save_gradient_ratio(unlearn_trainloader, self.model, optimizer,'/projectnb/ivc-ml/yuwentan/Unlearning/checkpoint/Tara/Unlearning/SALUN/')
                mask=torch.load('/projectnb/ivc-ml/yuwentan/Unlearning/checkpoint/Tara/Unlearning/SALUN/difficult_with_0.1.pt')
                train_loss, train_coarse, train_fine,iterations=Unlearn_training(self.model, self.originmodel, unlearn_trainloader,unlearn_testloader, optimizer, scheduler, epoch,args,log_interval,iterations,mask)
            else:
                mask=None
                train_loss,  train_coarse, train_fine,iterations =Unlearn_training(self.model, self.originmodel, unlearn_trainloader,unlearn_testloader, optimizer, scheduler, epoch,args,log_interval,iterations,mask)
            test_coarse_loss,test_fine_loss, test_coarse,test_fine = test(self.model, unlearn_testloader, epoch, args)
            wandb.log({
                "epoch":epoch,
                "forget_train/epoch_train_loss": train_loss,
                "forget_train/epoch_coarse_acc": train_coarse,
                "forget_train/epoch_fine_acc": train_fine,
                "forget_test/epoch_coarse_accuracy": test_coarse,
                "forget_test/epoch_fine_accuracy": test_fine
            })
            if args.unlearning_method=='neg' or args.unlearning_method=='Relabeling'or args.unlearning_method=='SALUN':
                if args.unlearning_goal=='fine':
                    if (train_fine * 100) >= self.trlog['fine_max_acc']:
                        self.trlog['fine_max_acc'] = float('%.3f' % (train_fine * 100))
                        self.trlog['fine_max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'difficult_fine_min_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        print('********A better model is found!!**********')
                        #print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],  self.trlog['max_acc']))
                if  args.unlearning_goal=='coarse':
                    if (train_coarse * 100) >= self.trlog['coarse_max_acc']:
                        self.trlog['coarse_max_acc'] = float('%.3f' % (coarse_fine * 100))
                        self.trlog['coarse_max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'coarse_min_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        print('********A better model is found!!**********')
                        #print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],  self.trlog['max_acc']))
            else:
                if args.unlearning_goal=='fine':
                    if (train_fine * 100) < self.trlog['fine_min_acc']:
                        self.trlog['fine_min_acc'] = float('%.3f' % (train_fine * 100))
                        self.trlog['min_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, f"150_difficult_KLc_{args.KL_c}_KLf={args.KL_f}_Margin_{args.margin_f}_beta_{args.beta}.pth")
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)   
                if  args.unlearning_goal=='coarse':
                    if (train_coarse * 100) <self.trlog['min_acc'] :
                        self.trlog['min_acc'] = float('%.3f' % (train_fine * 100))
                        self.trlog['min_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, f"KLc_{args.KL_c}_KLf={args.KL_f}_Margin_{args.margin_c}.pth")
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
            scheduler.step()
        save_model_dir = os.path.join(args.save_path, f"150_difficult_final_parameter_margin_{args.margin_f}_beta_{args.beta}.pth")
        torch.save(dict(params=self.model.state_dict()), save_model_dir)

    def set_save_path(self):
        self.args.save_path = '%s/' % self.args.unlearn_dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project+'%s/' % self.args.unlearning_method+'%s/' % self.args.unlearning_goal

        self.args.save_path = self.args.save_path+'/'
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.5f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs, self.args.lr, mile_stone, self.args.gamma, self.args.batch_size,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.5f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs, self.args.lr, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Cosine':
            self.args.save_path = self.args.save_path + 'Cosine-Epo_%d-Lr_%.8f' % (
                self.args.epochs, self.args.lr)


        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
