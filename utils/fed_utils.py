import torch
import copy
from prettytable import PrettyTable
import numpy as np
from torch.nn import functional as F

def show_results(cfg, results, epoch,global_test_acc_dict):

    global_test_acc = []
    global_test_error = []
    global_test_f1 = []
    for k, result in enumerate(results):
        global_test_acc.append(results[k]['accuracy'])
        global_test_error.append(results[k]['error_rate'])
        global_test_f1.append(results[k]['macro_f1'])

        if k in global_test_acc_dict:
            global_test_acc_dict[k].append(results[k]['accuracy'])
        else:
            global_test_acc_dict[k] = [results[k]['accuracy']]

        print(k, "--Local test acc:", results[k]['accuracy'])

    print("--Global test acc:", sum(global_test_acc) / len(global_test_acc))

    print(f"Epoch:{epoch}")
    return global_test_acc,global_test_acc_dict

def average_weights(w, idxs_users, datanumber_client, islist=False):
    """
    Returns the average of the weights.
    """
    total_data_points = sum([datanumber_client[r] for r in idxs_users])

    w_avg = copy.deepcopy(w[idxs_users[0]])
    for idx in range(len(idxs_users)):
        fed_avg_freqs = datanumber_client[idxs_users[idx]] / total_data_points

        if islist:
            if idx == 0:
                w_avg = w_avg * fed_avg_freqs
            else:
                w_avg += w[idxs_users[idx]] * fed_avg_freqs
        else:
            if idx == 0:
                for key in w_avg:
                    w_avg[key] = w_avg[key] * fed_avg_freqs
            else:
                for key in w_avg:
                    w_avg[key] += w[idxs_users[idx]][key] * fed_avg_freqs

    return w_avg

def moment_aggre_weights(w_prompt,w_key, global_key, global_group_prompt,cluster_size ,idxs_users, datanumber_client, moment = 0.5):
    
    total_data_points = sum([datanumber_client[r] for r in idxs_users])
    total_cluster_size = sum([cluster_size[r] for r in idxs_users])

    group_ratio = [[0 for r in range(global_key.shape[0])] for i in idxs_users]

    for idx in range(len(idxs_users)):
        for r in range(global_key.shape[0]):
            if total_cluster_size[r] <= 50:
                group_ratio[idx][r] = 1/len(idxs_users)
            else:
                group_ratio[idx][r] =  (cluster_size[idx][r] / total_cluster_size[[r]]).item()

    for idx in range(len(idxs_users)):
        fed_avg_freqs = datanumber_client[idxs_users[idx]] / total_data_points

        if idx == 0:
            unique_key = copy.deepcopy(global_key)
            unique_group_prompt = copy.deepcopy(global_group_prompt)
            global_group_prompt = w_prompt[idx] * fed_avg_freqs
            for i, s in enumerate(group_ratio[idx]):
                global_key[i] = w_key[idx][i] * s
        else:
            global_group_prompt += w_prompt[idx] * fed_avg_freqs
            for i, s in enumerate(group_ratio[idx]):
                global_key[i] += w_key[idx][i] * s

    moment_key_para = moment * unique_key + (1 - moment) * global_key
    moment_prompt_para = moment * unique_group_prompt + (1 - moment) * global_group_prompt
    return moment_prompt_para, moment_key_para


def count_parameters(model, model_name):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if model_name in name:
            # if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params



import os

def save_acc_csv(para_dir,global_test_acc_dict,cfg):
    acc_path = os.path.join(para_dir, 'acc.csv')
    if os.path.exists(acc_path):
        with open(acc_path, 'a') as result_file:
            for key in global_test_acc_dict:
                method_result = global_test_acc_dict[key]
                result_file.write(str(key) + ',')
                for epoch in range(len(method_result)):
                    result_file.write(str(method_result[epoch]))
                    if epoch != len(method_result) - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')
    else:
        with open(acc_path, 'w') as result_file:
            result_file.write('idx,')
            for epoch in range(cfg.OPTIM.ROUND):
                result_file.write('epoch_' + str(epoch))
                if epoch != cfg.OPTIM.ROUND - 1:
                    result_file.write(',')
                else:
                    result_file.write('\n')

            for key in global_test_acc_dict:
                method_result = global_test_acc_dict[key]
                result_file.write(str(key) + ',')
                for epoch in range(len(method_result)):
                    result_file.write(str(method_result[epoch]))
                    if epoch != len(method_result) - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')
            
   
class KMEANS:
    def __init__(self, n_clusters=2, max_iter=50, verbose=True,device = torch.device("cpu")):
        self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit_predict(self, x):
        # 随机选择初始中心点
        init_row = torch.randperm(x.shape[0])[:self.n_clusters].to(self.device)
        init_points = x[init_row]
        self.centers = init_points
        while True:
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmax(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 or self.max_iter == self.count:
                break
            self.count += 1

        return self.labels

    def nearest_center(self, x): 
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        for i, sample in enumerate(x):
            dist = F.cosine_similarity(sample.unsqueeze(0), self.centers,dim=1)
            labels[i] = torch.argmax(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels = labels
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers
