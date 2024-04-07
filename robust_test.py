# -*- coding: utf-8 -*-
# @Time    : 2023/8/28 19:56
# @Author  : pcy
# @Site    : 
# @File    : robust_test.py
# @Software: PyCharm 
# @Comment :

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso
import community.community_louvain as louvain
import itertools
from scipy import stats
from scipy.stats import pearsonr, norm

# 无参随机抽取
def random_sampling(pd_data, num_samples, sample_size):
    sample_list = []
    for i in range(num_samples):
        sample = pd_data.sample(frac=sample_size, replace=False, random_state=i)
        sample_list.append(sample)
    return sample_list


def matrix_preprocess(matrix):
    number_of_nodes = matrix.shape[1]
    matrix_result = matrix
    # 去对角线
    for i in range(0,number_of_nodes):
        matrix_result[i,i] = 0
    # 取绝对值
    matrix_result = abs(matrix_result)
    return matrix_result


def create_network(sample_list):
    return_list = []
    network_index = 1
    for sample_data in sample_list:
        # 计算网络模型，返回矩阵
        estimator = GraphicalLasso(alpha=0.05)
        estimator.fit(sample_data)

        #### 精准矩阵要进行处理，从逆协方差变成偏相关
        # 将精度矩阵对角线元素开根号
        diag_sqrt = np.sqrt(np.diag(estimator.precision_))
        # 计算偏相关矩阵
        partial_corr_matrix = -estimator.precision_ / np.outer(diag_sqrt, diag_sqrt)
        # 将对角线元素设置为1
        np.fill_diagonal(partial_corr_matrix, 1)

        # 生成nxG，输出网络基本信息、拓扑信息
        matrix = matrix_preprocess(partial_corr_matrix)
        nxG = nx.Graph(matrix, weight=True)
        nxG1 = nxG.copy()
        print("create network "+str(network_index)+" nodes:"+str(nxG1.number_of_nodes())+" edges:"+str(nxG1.number_of_edges()))
        return_list.append(nxG1)
        nxG.clear()
        network_index = network_index + 1

    return return_list


def CIs(data,alpha):
    # 置信区间选取95%
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    ci = stats.t.interval(alpha=alpha, df=len(data) - 1, loc=mean, scale=std)
    # print("95% CI:", ci)
    return ci, mean


def matrix_list_CIs(matrix_list,alpha,save_path):
    matrix_shape = matrix_list[0].shape
    node_num = matrix_list[0].shape[0]
    with open(save_path+"robust.txt", 'a') as file_object:
        file_object.write("source\t" +
                          "target\t" +
                          "confidence_interval\t" +
                          "mean\n"
                          )
    file_object.close()

    for i in range(0,node_num):
        for j in range(0,node_num):
            edge_list = []
            for matrix in matrix_list:
                edge_list.append(matrix[i,j])
            ci, mean = CIs(edge_list,alpha)
            with open(save_path+"robust.txt", 'a') as file_object:
                file_object.write(str(i) + "\t" + str(j) + "\t" + str(ci)+ "\t" + str(mean)+ "\n")
            file_object.close()
    return None


def edge_robust_test(data_path,save_path,num_samples,sample_size,alpha):
    # read data
    # 读取csv,生成pd
    pd_data = pd.read_csv(data_path)
    ## 检验是否有缺省值
    print(np.isnan(pd_data).any())
    sample_list = random_sampling(pd_data, num_samples, sample_size)
    # 重复建网
    nx_list = create_network(sample_list)
    # 计算检验指标, 这里强度不使用归一化的值，在后续比较的过程中，真实网络也应使用原始值
    # 连边稳定性检验,保存连边数
    with open(save_path+"edge_num_bootstrapping.txt", 'a') as file_object:
        file_object.write(
                          "edge_num\n"
                          )
    file_object.close()
    edge_matrix_list = []
    # 累加矩阵
    edge_matrix_sum = np.zeros((nx_list[0].number_of_nodes(), nx_list[0].number_of_nodes()))
    for nxG in nx_list:
        # 单一网络的具体连边矩阵
        edge_matrix_list.append(nx.adjacency_matrix(nxG).todense())
        with open(save_path + "edge_num_bootstrapping.txt", 'a') as file_object:
            file_object.write(
                str(nxG.number_of_edges())+"\n"
            )
        file_object.close()
        # 累加连边数量
        for edge in nxG.edges:
            source = edge[0]
            target = edge[1]
            edge_matrix_sum[source,target] = edge_matrix_sum[source,target] + 1
            edge_matrix_sum[target,source] = edge_matrix_sum[target,source] + 1

    # CIs,计算连边稳定性，以及网络结构稳定性（在最终结果中，均值为0的边始终没有出现）
    matrix_list_CIs(edge_matrix_list, alpha, save_path)
    np.savetxt(save_path+"edge_matrix.txt",edge_matrix_sum/num_samples)
    return None


if __name__ == '__main__':

    # source data
    data_path = "D:\\pycharm\\workspace\\brainNetwork_pcy_test\\symptom_network_juan\\data\\T0self141items_all_tocy.csv"
    save_path = "D:\\pycharm\\workspace\\brainNetwork_pcy_test\\symptom_network_juan\\result\\"

    # 随机抽样次数
    num_samples = 1000
    # 每次抽取的比例
    sample_size = 0.9
    # 置信区间
    alpha = 0.95
    # 采用进行连边的鲁棒性分析,随机抽取1000次，每次抽取全样本80%
    edge_robust_test(data_path, save_path, num_samples, sample_size, alpha)

