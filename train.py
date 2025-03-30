import os
import argparse
import importlib
from utils import *


def get_command_line_parser():
    parser = argparse.ArgumentParser()
    # setting of the dataset and the unlearning task
    parser.add_argument('-unlearn_dataset', type=str, default='Subsetdog',  choices=['Compcars','Subsetdog'])
    parser.add_argument('-template', type=str, default='Subsetdog_test',  choices=['Compcars_train','Compcars_test','Subsetdog_test','Subsetdog_train'])
    parser.add_argument('-unlearn_fine_classes',type=str,default=['Border terrier', 'Irish terrier', 'wire-haired fox terrier', 'Sealyham terrier', 'cairn terrier', 'Scottish terrier'],
    nargs='+', help='List of fine classes to unlearn.')                    
    parser.add_argument('-unlearn_coarse_classes',type=str,default=['Terrier'],nargs='+', help='List of coarse classes to unlearn.')
    # about pre-training
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-epochs', type=int, default=8)
    parser.add_argument('-backbonename', type=str, default='ViT-L/14', choices=['ViT-L/14','ViT-B/16'])
    parser.add_argument('-lr', type=float, default=1e-7)
    parser.add_argument('-schedule', type=str, default='Cosine',choices=['Step', 'Milestone', 'Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=40)
    parser.add_argument('-decay', type=float, default=1e-5)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-batch_size_base', type=int, default=32)
    parser.add_argument('-test_batch_size', type=int, default=128)
    parser.add_argument('-model_dir', type=str, default=None, help='loading model parameter from a specific dir')
    # about training
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    #about unlearning medthods
    parser.add_argument('-unlearning_goal', type=str, default='fine', choices=['coarse','fine','hybrid'])
    parser.add_argument('-unlearning_method', type=str, default='Relabeling', choices=['GDiff','GA','Relabeling','KL','SALUN','ME','neg','NPO','Hinge'])
    parser.add_argument('-beta', type=float, default=0.4,help='hyper parameyer for NPO') 
    parser.add_argument('-KL_c', type=float, default=5,help='hyper parameyer of coarse KL divergence')
    parser.add_argument('-KL_f', type=float, default=20,help='hyper parameyer of fine KL divergence') 
    parser.add_argument('-margin_c', type=float,default=1,help='hyper parameyer for coasrse Hinge') 
    parser.add_argument('-margin_f', type=float,default=2,help='hyper parameyer for fine Hinge') 
    
    return parser


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    trainer = importlib.import_module('Unlearning.trainer' ).ULTrainer(args)

    trainer.train()