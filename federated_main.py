from collections import defaultdict

from utils.fed_utils import average_weights, count_parameters, show_results, save_acc_csv
from Dassl.dassl.utils import setup_logger, set_random_seed
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer
import setproctitle
import numpy as np
import argparse
import random
import torch
import time
import copy
import os


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg, args):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.PROMPTFL = CN()
    cfg.TRAINER.PROMPTFL.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.PROMPTFL.CSC = False  # class-specific context
    cfg.TRAINER.PROMPTFL.CTX_INIT = False  # initialization words
    cfg.TRAINER.PROMPTFL.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.GL_SVDMSE = CN()
    cfg.TRAINER.GL_SVDMSE.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.GL_SVDMSE.CSC = False  # class-specific context
    cfg.TRAINER.GL_SVDMSE.CTX_INIT = False  # initialization words
    cfg.TRAINER.GL_SVDMSE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.GL_SVDMSE.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.GL_SVDMSE.N = 1  # number of prompts
    cfg.TRAINER.GL_SVDMSE.lambda_orthogonal = 1
    cfg.TRAINER.GL_SVDMSE.alpha = args.alpha
    cfg.TRAINER.GL_SVDMSE.ratio = args.ratio
    
    cfg.TRAINER.GL_SVDMSE_HE = CN()
    cfg.TRAINER.GL_SVDMSE_HE.N_CTX_GLOBAL = args.n_ctx  # number of context vectors
    cfg.TRAINER.GL_SVDMSE_HE.CSC = False  # class-specific context
    cfg.TRAINER.GL_SVDMSE_HE.CTX_INIT = False  # initialization words
    cfg.TRAINER.GL_SVDMSE_HE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.GL_SVDMSE_HE.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.GL_SVDMSE_HE.N = 1  # number of prompts
    cfg.TRAINER.GL_SVDMSE_HE.lambda_orthogonal = 1
    cfg.TRAINER.GL_SVDMSE_HE.alpha = args.alpha
    cfg.TRAINER.GL_SVDMSE_HE.ratio = args.ratio
    
    cfg.TRAINER.GLP_OT = CN()
    cfg.TRAINER.GLP_OT.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.GLP_OT.CSC = False  # class-specific context
    cfg.TRAINER.GLP_OT.CTX_INIT = False  # initialization words
    cfg.TRAINER.GLP_OT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.GLP_OT.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.GLP_OT.N = args.num_prompt  # number of prompts
    cfg.TRAINER.GLP_OT.AVG_N = args.num_prompt / 2  # number of prompts to aggregate
    cfg.TRAINER.GLP_OT.THRESH = 1e-3  # thresh of sinkhorn distance
    cfg.TRAINER.GLP_OT.EPS = 0.1  # lambada of sinkhorn distance
    cfg.TRAINER.GLP_OT.OT = 'COT'  # type of OT used Sinkhorn(for standard OT), COT(for unbalanced OT)")
    cfg.TRAINER.GLP_OT.TOP_PERCENT = 1
    cfg.TRAINER.GLP_OT.MAX_ITER = 100
    
    cfg.TRAINER.FEDPGP = CN()
    cfg.TRAINER.FEDPGP.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.FEDPGP.CSC = False  # class-specific context
    cfg.TRAINER.FEDPGP.CTX_INIT = False  # initialization words
    cfg.TRAINER.FEDPGP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.FEDPGP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.FEDPGP.BOTTLENECK = 4
    cfg.TRAINER.FEDPGP.N = 1 # number of prompts
    cfg.TRAINER.FEDPGP.FEATURE = False
    cfg.TRAINER.FEDPGP.mu = 1
    cfg.TRAINER.FEDPGP.temp = 0.5

    # current_trainer_cfg = cfg['TRAINER'][args.trainer]
    # cfg['TRAINER'] = CN()
    # cfg['TRAINER'][args.trainer] = current_trainer_cfg
    
    # current_trainer_cfg = cfg['TRAINER'][args.trainer]
    
    # cfg['TRAINER'] = CN(cfg['TRAINER'])
    # cfg['TRAINER'][args.trainer] = current_trainer_cfg

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.USERS = args.num_users  # number of clients
    cfg.DATASET.NAME = args.dataset
    
    cfg.DATASET.USER_PROMPT_LENGTHS = []
    
    cfg.DATASET.IID = args.iid  # is iid
    cfg.DATASET.PARTITION = args.partition

    cfg.DATASET.USEALL = args.useall  # use all data for training instead of few shot
    cfg.DATASET.NUM_SHOTS = args.num_shots

    cfg.DATASET.BETA = args.beta
    cfg.DATASET.REPEATRATE = 0.0  # repeat rate on each client

    cfg.OPTIM.ROUND = 1  # global round
    cfg.OPTIM.GAMMA = args.gamma  # gamma of single-step

    cfg.MODEL.BACKBONE.PRETRAINED = True


def setup_cfg(args):
    cfg = get_cfg_default()

    extend_cfg(cfg, args)

    # 1. From the dataset config file
    if args.dataset:
        cfg.merge_from_file(f'configs/datasets/{args.dataset}.yaml')

    # 2. set batch size
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size
    
    # 3. From input arguments
    reset_cfg(cfg, args)
    
    random.seed(cfg.SEED)
    if cfg.DATASET.NAME.lower() in ["cifar10", "cifar100"]:
        cfg.DATASET.USER_PROMPT_LENGTHS = [
            random.randint(4, 32) for _ in range(cfg.DATASET.USERS)
        ]

    # 数据集逻辑
    if cfg.DATASET.NAME in ["cifar10", "cifar100"]:
        cfg.DATASET.USER_PROMPT_LENGTHS = [
            random.randint(4, 32) for _ in range(cfg.DATASET.USERS)
        ]
    elif cfg.DATASET.NAME in ["Office31", "OfficeHome"]:  
        if args.specify:
            if args.prompts_lens is None or len(args.prompts_lens) != cfg.DATASET.USERS:
                raise ValueError(
                    "When using --specify, you must provide a --prompts_lens list "
                    "with the same number of elements as the number of users."
                )
            cfg.DATASET.USER_PROMPT_LENGTHS = args.prompts_lens

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)
    
    cfg.OUTPUT_DIR = f"output/{args.dataset}/{args.trainer}/shot_{args.num_shots}/beta_{args.beta}/ep{cfg.OPTIM.MAX_EPOCH}_r{cfg.OPTIM.ROUND}/alpha{args.alpha}_ratio{args.ratio}/seed_{args.seed}"
    
    if args.specify and cfg.DATASET.NAME.lower() == "office31":
        prompts_lens_str = "_".join(map(str, args.prompts_lens))
        cfg.OUTPUT_DIR = (
            f"output/{args.dataset}/{args.trainer}/"
            f"specify_{args.specify}/beta_{args.beta}/"
            f"ep{cfg.OPTIM.MAX_EPOCH}_r{cfg.OPTIM.ROUND}/"
            f"alpha{args.alpha}_ratio{args.ratio}/"
            f"/prompts_{prompts_lens_str}/"
            f"seed_{args.seed}"
        )
    
    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    args.para_dir = setup_logger(cfg)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    
    local_weights_0 = [[] for i in range(cfg.DATASET.USERS)]
    local_weights_1 = [[] for i in range(cfg.DATASET.USERS)]
    local_weights_2 = [[] for i in range(cfg.DATASET.USERS)]
    local_weights_3 = [[] for i in range(cfg.DATASET.USERS)]
    local_weights_per = [{} for i in range(cfg.DATASET.USERS)]

    local_trainer = build_trainer(args, cfg)

    local_trainer.fed_before_train()
    count_parameters(local_trainer.model, "prompt_learner")
    count_parameters(local_trainer.model, "image_encoder")
    count_parameters(local_trainer.model, "text_encoder")

    datanumber_client = []
    if args.trainer == 'CLIP':
        global_weights = copy.deepcopy(local_trainer.model.state_dict())
    else:
        for net_i in range(cfg.DATASET.USERS):
            # local_trainer = build_trainer(cfg)
            datanumber_client.append(len(local_trainer.fed_train_loader_x_dict[net_i].dataset))
        global_weights = copy.deepcopy(local_trainer.model.state_dict())

    # Training
    start_epoch = 0
    end_epoch = cfg.OPTIM.ROUND
    global_test_acc_dict = {}
    global_time_list = []
    start = time.time()
    for epoch in range(start_epoch, end_epoch):

        if args.trainer == 'CLIP':
            print("------------Global test start (不训练) -------------")
            results = []

            idxs_users = list(range(cfg.DATASET.USERS))
            print("idxs_users:", idxs_users)

            for idx in idxs_users:
                local_trainer.model.load_state_dict(global_weights, strict=False)
                
                result = local_trainer.test(idx=idx)
                results.append(result)

            global_test_acc, global_test_acc_dict = show_results(cfg, results, epoch, global_test_acc_dict)
            global_time_list.append(time.time() - start)

            print("------------Global test finish-------------")

        elif args.trainer == 'PROMPTFL':
            # global prompt + local prompt

            idxs_users = list(range(0, cfg.DATASET.USERS))
            print("idxs_users", idxs_users)

            print("------------local train start epoch:", epoch, "-------------")
            for idx in idxs_users:
                if epoch == 0:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                else:
                    local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                local_weight = local_trainer.model.state_dict()
                local_weights_0[idx] = copy.deepcopy(local_weight['prompt_learner.ctx'])

            print("------------local train finish epoch:", epoch, "-------------")

            global_weights = average_weights(local_weights_0, idxs_users, datanumber_client, islist=True)

            print("------------local test start-------------")
            results = []
            all_users = list(range(0, cfg.DATASET.USERS))

            for idx in all_users:
                local_weights_per[idx]['prompt_learner.ctx'] = global_weights

            for idx in all_users:
                local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                results.append(local_trainer.test(idx=idx))
            # global_test_acc = show_results(cfg, results, epoch)
            global_test_acc, global_test_acc_dict = show_results(cfg, results, epoch, global_test_acc_dict)
            global_time_list.append(time.time() - start)
            print("------------local test finish-------------")
            
        elif args.trainer == 'GL_SVDMSE':
            # global prompt + local prompt

            idxs_users = list(range(0, cfg.DATASET.USERS))
            print("idxs_users", idxs_users)

            print("------------local train start epoch:", epoch, "-------------")
            for idx in idxs_users:
                if epoch == 0:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                else:
                    local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                local_weight = local_trainer.model.state_dict()
                local_weights_0[idx] = copy.deepcopy(local_weight['prompt_learner.ctx_global'])
                local_weights_1[idx] = copy.deepcopy(local_weight['prompt_learner.ctx_local'])

            print("------------local train finish epoch:", epoch, "-------------")

            global_weights = average_weights(local_weights_0, idxs_users, datanumber_client, islist=True)

            print("------------local test start-------------")
            results = []
            all_users = list(range(0, cfg.DATASET.USERS))

            for idx in all_users:
                local_weights_per[idx]['prompt_learner.ctx_global'] = global_weights
                local_weights_per[idx]['prompt_learner.ctx_local'] = local_weights_1[idx]

            for idx in all_users:
                local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                results.append(local_trainer.test(idx=idx))
            # global_test_acc = show_results(cfg, results, epoch)
            global_test_acc, global_test_acc_dict = show_results(cfg, results, epoch, global_test_acc_dict)
            global_time_list.append(time.time() - start)
            print("------------local test finish-------------")
            
        elif args.trainer == 'GL_SVDMSE_HE':
            # global prompt + local prompt

            idxs_users = list(range(0, cfg.DATASET.USERS))
            print("idxs_users", idxs_users)

            print("------------local train start epoch:", epoch, "-------------")
            for idx in idxs_users:
                if epoch == 0:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                else:
                    local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                local_weight = local_trainer.model.state_dict()
                local_weights_0[idx] = copy.deepcopy(local_weight['prompt_learner.ctx_global'])
                local_weights_1[idx] = {}
                for i, param_name in enumerate([f'prompt_learner.ctx_local_list.{i}' for i in range(len(local_trainer.model.prompt_learner.ctx_local_list))]):
                    local_weights_1[idx][i] = copy.deepcopy(local_weight[param_name])

            print("------------local train finish epoch:", epoch, "-------------")

            global_weights = average_weights(local_weights_0, idxs_users, datanumber_client, islist=True)

            print("------------local test start-------------")
            results = []
            all_users = list(range(0, cfg.DATASET.USERS))

            for idx in all_users:
                local_weights_per[idx]['prompt_learner.ctx_global'] = global_weights
                for i, param_name in enumerate([f'prompt_learner.ctx_local_list.{i}' for i in range(len(local_trainer.model.prompt_learner.ctx_local_list))]):
                    local_weights_per[idx][param_name] = local_weights_1[idx][i]

            for idx in all_users:
                local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                results.append(local_trainer.test(idx=idx))
            # global_test_acc = show_results(cfg, results, epoch)
            global_test_acc, global_test_acc_dict = show_results(cfg, results, epoch, global_test_acc_dict)
            global_time_list.append(time.time() - start)
            print("------------local test finish-------------")
                  
        elif args.trainer == 'GLP_OT':
            # global prompt + local prompt

            idxs_users = list(range(0, cfg.DATASET.USERS))
            print("idxs_users", idxs_users)

            print("------------local train start epoch:", epoch, "-------------")
            for idx in idxs_users:
                if epoch == 0:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                else:
                    local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                local_weight = local_trainer.model.state_dict()
                local_weights_0[idx] = copy.deepcopy(local_weight['prompt_learner.ctx'][:args.avg_prompt])
                local_weights_1[idx] = copy.deepcopy(local_weight['prompt_learner.ctx'][args.avg_prompt:args.num_prompt])

            print("------------local train finish epoch:", epoch, "-------------")

            global_weights = average_weights(local_weights_0, idxs_users, datanumber_client, islist=True)

            print("------------local test start-------------")
            results = []
            all_users = list(range(0, cfg.DATASET.USERS))

            for idx in all_users:
                local_weights_per[idx]['prompt_learner.ctx'] = torch.cat([global_weights, local_weights_1[idx]], dim=0)

            for idx in all_users:
                local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                results.append(local_trainer.test(idx=idx))
            # global_test_acc = show_results(cfg, results, epoch)
            global_test_acc, global_test_acc_dict = show_results(cfg, results, epoch, global_test_acc_dict)
            global_time_list.append(time.time() - start)
            print("------------local test finish-------------")

        elif args.trainer == 'FEDPGP':
            # Reparameterization prompt for personal FL
            idxs_users = list(range(0, cfg.DATASET.USERS))

            print("idxs_users", idxs_users)
            print("------------local train start epoch:", epoch, "-------------")
            for idx in idxs_users:
                if epoch == 0:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                else:
                    local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                local_weight = local_trainer.model.state_dict()
                local_weights_0[idx] = copy.deepcopy(local_weight['prompt_learner.sigma'])
                local_weights_1[idx] = copy.deepcopy(local_weight['prompt_learner.U'])
                local_weights_2[idx] = copy.deepcopy(local_weight['prompt_learner.V'])
            print("------------local train finish epoch:", epoch, "-------------")

            global_weights = average_weights(local_weights_0, idxs_users, datanumber_client, islist=True)

            print("------------local test start-------------")
            results = []
            all_users = list(range(0, cfg.DATASET.USERS))

            for idx in all_users:
                local_weights_per[idx]['prompt_learner.sigma'] = global_weights
                local_weights_per[idx]['prompt_learner.U'] = local_weights_1[idx]
                local_weights_per[idx]['prompt_learner.V'] = local_weights_2[idx]
            
            for idx in all_users:
                local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                results.append(local_trainer.test(idx=idx))

            # global_test_acc = show_results(cfg, results, epoch)
            global_test_acc, global_test_acc_dict = show_results(cfg, results, epoch, global_test_acc_dict)
            global_time_list.append(time.time() - start)
            print("------------local test finish-------------")

    for idx in idxs_users:
        local_trainer.fed_after_train()
    for key, global_test_acc_list in global_test_acc_dict.items():
        print(key, "global_test_acc_list:", global_test_acc_list)
        print(key, "maximum test acc:", max(global_test_acc_list))
        print(key, "mean of acc:", np.mean(global_test_acc_list[-5:]))
        print(key, "std of acc:", np.std(global_test_acc_list[-5:]))
    save_acc_csv(local_trainer.args.para_dir, global_test_acc_dict, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=str, default="PROMPTFL_PROX", help="name of trainer, choose from: "
                                                                      "Baseline, CLIP, FEDPGP, GLP_OT, FEDPEFT")
    parser.add_argument("--dataset", type=str, default="dtd", help="name of dataset, choose from: "
                                                                    " cifar100 domainnet pacs OfficeHome  Office31 ")
    parser.add_argument("--backbone", type=str, default="ViT-B/16", help="name of CNN backbone")
    parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution')

    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help='the fraction of clients: C')
    parser.add_argument('--gamma', type=float, default=1, help='gamma of single_step')
    parser.add_argument('--train_batch_size', type=int, default=32, help="number of trainer batch size")
    parser.add_argument('--test_batch_size', type=int, default=128, help="number of test batch size")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")

    # parameters of datasets
    # caltech101, oxford_flowers, oxford_pets, food101 and dtd
    parser.add_argument('--num_shots', type=int, default=2, help="number of shots in few shot setting")
    parser.add_argument('--useall', default=False, help="is useall, True for all training samples, False for few shot learning")
    parser.add_argument('--iid', default=False, help="is iid, control the iid of caltech101, oxford_flowers, oxford_pets, food101 and dtd")

    # cifar10, cifar100
    parser.add_argument('--partition', type=str, default='noniid-labeldir',
                        help='the data partitioning strategy of cifar10 and cifar100,'
                             ' select from "noniid-labeluni, noniid-labeldir,noniid-labeldir100"')

    # parameters of learnable prompts
    parser.add_argument('--n_ctx', type=int, default=16, help="number of text encoder of text prompts")
    parser.add_argument('--num_prompt', type=int, default=2, help="number of prompts")
    parser.add_argument('--avg_prompt', type=int, default=1, help="half number of prompts")
    # FedPHA
    parser.add_argument('--alpha', type=float, default=1.0, help="The parameter for push_loss")
    parser.add_argument('--ratio', type=float, default=0.8, help="The parameter for svd")
    # he setting
    parser.add_argument('--specify', default=False, help="Whether to specify the prompt length list of the dataset")
    parser.add_argument('--prompts_lens', nargs='+', type=int, help="Specify the prompt length list of the dataset, eg.--prompts_lens 4 8 16 32")
    
    # parameters of path
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument("--root", type=str, default="/data/fcy_data", help="path to dataset")
    parser.add_argument("--output_dir", type=str, default="output/..", help="output directory")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    # parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")
    
    args = parser.parse_args()
    
    setproctitle.setproctitle('{}_{}_{}'.format(args.trainer, args.backbone, args.dataset))

    main(args)
