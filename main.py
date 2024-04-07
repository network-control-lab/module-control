# -*- coding: utf-8 -*-
# @Time    : 2023/6/2 10:49
# @Author  : pcy
# @Site    : 
# @File    : main.py
# @Software: PyCharm 
# @Comment : 

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.covariance import GraphicalLasso
from sklearn.covariance import GraphicalLassoCV
import community.community_louvain as louvain
import random
import itertools
from scipy import stats
import datetime

def all_min_dominating_set(nxG):
    min_dominating_set_result = []
    strength_of_all_min_dominating_set = []
    # nxG中的节点非重复且不考虑顺序的排列组合,暴力求解最小支配集
    node_list = nx.nodes(nxG)
    node_num = nxG.number_of_nodes()
    # 初始化最小支配集大小为所有节点
    min_dominating_size = node_num
    print("start searching....")
    for i in range(1,node_num+1):
        print("searching for size " + str(i) +" ...")
        print(str(datetime.datetime.now()))
        set_list = list(itertools.combinations(node_list, i))
        for temp_set in set_list:
            if nx.is_dominating_set(nxG,temp_set):
                min_dominating_size = i
                min_dominating_set_result.append(temp_set)
                # 计算最小支配集权重
                temp_total_strength = 0
                for dominating_node in temp_set:
                    temp_strength = nxG.degree(dominating_node,weight='weight')
                    temp_total_strength = temp_total_strength + temp_strength
                strength_of_all_min_dominating_set.append(temp_total_strength)
        # 如果已经找到最小支配集，那么就不再继续寻找最小支配集
        if i >= min_dominating_size:
            break
        print(str(datetime.datetime.now()))
    return min_dominating_set_result,strength_of_all_min_dominating_set

def dominating_frequency(all_dom_set,nxG):
    num_dom_set = len(all_dom_set)
    node_num = nxG.number_of_nodes()
    # init
    as_dom_node_count = {}
    for node_index in range(0,node_num):
        as_dom_node_count[node_index] = 0
    # count
    for min_dom_set in all_dom_set:
        for dom_node in min_dom_set:
            as_dom_node_count[dom_node] = as_dom_node_count[dom_node] + 1

    for node_index in as_dom_node_count:
        as_dom_node_count[node_index] = as_dom_node_count[node_index] / num_dom_set
    print(as_dom_node_count)
    return as_dom_node_count

def matrix_preprocess(matrix):
    number_of_nodes = matrix.shape[1]
    matrix_result = matrix
    # 去对角线
    for i in range(0,number_of_nodes):
        matrix_result[i,i] = 0
    # 取绝对值
    matrix_result = abs(matrix_result)
    return matrix_result

def module_controllability(nxG,all_dom_set,louvain_communities):

    # louvain_communities = louvain.best_partition(nxG)
    number_of_communities = max(louvain_communities.values())+1
    print("module number: "+str(number_of_communities))
    # init
    module = {}
    for index in range(0,number_of_communities):
        module[index] = []
    # finding module
    for node in louvain_communities:
        community_index = louvain_communities[node]
        module[community_index].append(node)

    for i in range(0,number_of_communities):
        print("module "+str(i)+" has "+ str(module[i])+" nodes")

    # 初始化结果
    average_module_controllability_result = {}
    for module_source in module:
        for module_target in module:
            average_module_controllability_result[str(module_source) + "_" + str(module_target)] = 0


    # all_dom_set,strength_of_all_dom_set = all_min_dominating_set(nxG)
    # print(all_dom_set)
    for min_dom_set in all_dom_set:
        dominated_area = {}
        for dom_node in min_dom_set:
            temp_neighbor = set()
            temp_neighbor.clear()
            for neighbor in nxG.neighbors(dom_node):
                temp_neighbor.add(neighbor)
            #支配域还有节点自身
            temp_neighbor.add(dom_node)
            dominated_area[dom_node] = temp_neighbor

        # 计算社团支配域
        modules_control_area = {}
        for module_index in module:
            node_in_module = module[module_index]
            single_module_control_area = set()
            for node in node_in_module:
                if node in min_dom_set:
                    # 添加module支配域
                    temp_dom_set = set()
                    for temp_node in dominated_area[node]:
                        single_module_control_area.add(temp_node)
            modules_control_area[module_index] = single_module_control_area
        # print(modules_control_area)

        # 计算社团间支配能力
        temp_module_controllability_result = {}
        temp_module_controllability_result.clear()
        for module_source in module:
            for module_target in module:
                # 社团控制域
                control_area = modules_control_area[module_source]
                # 被控社团节点集
                target_module_area = module[module_target]
                # 两者交集大小
                target_module_area_set = set(target_module_area)
                inter = control_area.intersection(target_module_area_set)
                temp_module_controllability_result[str(module_source)+"_"+str(module_target)] = len(inter) / len(target_module_area)
                average_module_controllability_result[str(module_source) + "_" + str(module_target)] = average_module_controllability_result[str(module_source) + "_" + str(module_target)] + (len(inter) / len(target_module_area))
        print("dom_set: "+str(min_dom_set) + "   module_controllability: "+ str(temp_module_controllability_result))
    for total_module_controllability in average_module_controllability_result:
        average_module_controllability_result[total_module_controllability] = average_module_controllability_result[total_module_controllability] / len(all_dom_set)
    print("average_module_controllability: "+ str(average_module_controllability_result))
    return average_module_controllability_result

def network_analysis(nxG):
    network_analysis_result = {}
    # 聚集系数
    clustering = nx.clustering(nxG)
    # 接近中心性
    closeness = nx.closeness_centrality(nxG)
    # 介数中心性
    betweenness = nx.betweenness_centrality(nxG)
    # 度中心性（这里要补充进强度）
    degree = nx.degree_centrality(nxG)
    # 平均强度
    average_strength = {}
    print(nxG.nodes)
    for node in list(nxG.nodes):
        if nxG.degree(node) != 0:
            average_strength[node] = nxG.degree(node,weight='weight') / nxG.degree(node)
        else:
            average_strength[node] = 0
    # k core
    nG_nonself = nxG
    nG_nonself.remove_edges_from(nx.selfloop_edges(nxG))
    kcore = nx.core_number(nG_nonself)

    network_analysis_result["clustering"] = clustering
    network_analysis_result["closeness"] = closeness
    network_analysis_result["betweenness"] = betweenness
    network_analysis_result["degree"] = degree
    network_analysis_result["average_strength"] = average_strength
    network_analysis_result["kcore"] = kcore

    return network_analysis_result


def greedy_minimum_dominating_set(nxG, times):
    min_dominating_set = []

    for time in range(times):
        nxG_copy = nxG.copy()
        dominating_set = []

        while nxG_copy.nodes():
            node = random.choice(list(nxG_copy.nodes()))
            dominating_set.append(node)
            remove_list = []
            remove_list.clear()
            remove_list.append(node)
            for neighbor in nxG_copy.neighbors(node):
                remove_list.append(neighbor)

            for node in remove_list:
                nxG_copy.remove_node(node)

        dominating_set = set(dominating_set)
        if len(min_dominating_set) == 0:
            min_dominating_set.append(dominating_set)
        elif len(min_dominating_set[0]) == len(dominating_set) and dominating_set not in min_dominating_set:
            min_dominating_set.append(dominating_set)
        elif len(min_dominating_set[0]) > len(dominating_set):
            min_dominating_set.clear()
            min_dominating_set.append(dominating_set)

        print("times: " + str(time + 1) +" MDSet size: "+ str(len(min_dominating_set[0]))+ " MDSet number: "+ str(len(min_dominating_set)) +"  MDSet: " + str(min_dominating_set))

    return min_dominating_set


if __name__ == '__main__':
    ## 文件路径，要计算的数据的文件夹
    data_path = "D:\\pycharm\\workspace\\brainNetwork_pcy_test\\symptom_network_juan\\data\\"
    ## 文件路径，文件要保存到的文件夹
    result_path = "D:\\pycharm\\workspace\\brainNetwork_pcy_test\\symptom_network_juan\\result\\"
    ## 要计算的网络名字
    data_file = "T0self141items_all_tocy"

    # 读取csv,生成pd
    pd_data = pd.read_csv(data_path+data_file+".csv")
    print(pd_data)
    ## 检验是否有缺省值
    print(np.isnan(pd_data).any())

    # 计算网络模型，返回矩阵
    ################ 可选参数1 ###################
    ## GraphicalLassoCV()使用交叉验证自动确定alpha
    # estimator = GraphicalLassoCV()
    ## GraphicalLasso()需要给定参数alpha，默认alpha=0.01，alpha越大网络越稀疏
    estimator = GraphicalLasso(alpha=0.05)
    estimator.fit(pd_data)
    print(estimator.precision_)
    print(estimator.precision_.shape)

    #### 精准矩阵要进行处理，从逆协方差变成偏相关
    # 将精度矩阵对角线元素开根号
    diag_sqrt = np.sqrt(np.diag(estimator.precision_))
    # 计算偏相关矩阵
    partial_corr_matrix = -estimator.precision_ / np.outer(diag_sqrt, diag_sqrt)
    # 将对角线元素设置为1
    np.fill_diagonal(partial_corr_matrix, 1)
    print(partial_corr_matrix)

    # # 生成nxG，输出网络基本信息、拓扑信息
    matrix = matrix_preprocess(partial_corr_matrix)
    nxG = nx.Graph(matrix,weight=True)
    print(nxG.number_of_edges())
    network_analysis_result = network_analysis(nxG)

    ## network module
    louvain_communities = louvain.best_partition(nxG)
    number_of_communities = max(louvain_communities.values())+1

    # 计算minimum dominating sets
    ################ 可选参数2 ###################
    print(str(datetime.datetime.now()))
    # all_dom_set,strength_of_all_dom_set = all_min_dominating_set(nxG)
    # algorithm = "precise"
    all_dom_set = greedy_minimum_dominating_set(nxG, 1000)
    algorithm = "greedy"
    print(str(datetime.datetime.now()))
    print("all dom set: "+ str(all_dom_set))
    # print("dom set weight: "+str(strength_of_all_dom_set))

    # ACF
    as_dom_node_count = dominating_frequency(all_dom_set,nxG)
    print("frequency as dom nodes: "+str(as_dom_node_count))

    ## module controllability 这里必须传入louvain community的结果，不能在函数中算，社团编号可能会错乱
    average_module_controllability_result = module_controllability(nxG,all_dom_set,louvain_communities)

    nx.write_gexf(nxG, result_path+data_file+".gexf")


    # save result
    with open(result_path + data_file + ".txt", 'a') as file_object:
        file_object.write("item\t" +
                          "degree_centrality\t" +
                          "average_strength\t" +
                          "clustering\t" +
                          "closeness\t" +
                          "betweenness\t" +
                          "kcore\t" +
                          "module\t" +
                          "CF\n"
                          )
    file_object.close()

    with open(result_path + data_file + ".txt", 'a') as file_object:
        for node in nxG.nodes:
            file_object.write(str(node) + "\t" +
                              str(network_analysis_result["degree"][node]) + "\t" +
                              str(network_analysis_result["average_strength"][node]) + "\t" +
                              str(network_analysis_result["clustering"][node]) + "\t" +
                              str(network_analysis_result["closeness"][node]) + "\t" +
                              str(network_analysis_result["betweenness"][node]) + "\t" +
                              str(network_analysis_result["kcore"][node]) + "\t" +
                              str(louvain_communities[node]) + "\t" +
                              str(as_dom_node_count[node])+ "\n"
                              )
    file_object.close()

    # edge
    with open(result_path + data_file + ".txt", 'a') as file_object:
        file_object.write("source\t" +
                          "target\t" +
                          "weight\n"
                          )
    file_object.close()
    with open(result_path + data_file + ".txt", 'a') as file_object:
        for edge in nxG.edges:
            file_object.write(str(edge[0])+"\t" +
                              str(edge[1])+"\t" +
                              str(nxG.get_edge_data(edge[0],edge[1])['weight'])+"\n"
                              )
    file_object.close()


    # module controllability
    with open(result_path + data_file + ".txt", 'a') as file_object:
        file_object.write("module_2_module(direct)\t" +
                          "AMCS\n"
                          )
        for amcs in average_module_controllability_result:
            file_object.write(str(amcs)+"\t" +
                              str(average_module_controllability_result[amcs])+"\n"
                              )
    file_object.close()



    # minimum dominating set and its weight
    with open(result_path + data_file + ".txt", 'a') as file_object:

        file_object.write("minimum dominating set (" + str(algorithm)+ ") size: "+str(len(all_dom_set[0]))+"\n"
                          )
        for mds in all_dom_set:
            file_object.write(str(mds)+"\n"
                              )
    file_object.close()

