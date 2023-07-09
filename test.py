import time
from math import ceil
import numpy as np
from numpy import loadtxt
from sklearn.metrics.cluster import normalized_mutual_info_score
import json
import os

class Node(object):
    index = 0 # 当前节点编号
    neighbors = [] # 邻居节点列表
    degree = 1 # 节点的度
    max_similarity_neighbor = -1 # 最相似的邻接节点
    max_similarity = 0 # 最大相似度
    max_degree_neighbor = -1 # 度最大的邻接节点
    max_neighbor = -1 # 最大相似度=0，就是度最大的邻接节点，最大相似度≠0，就是最相似的节点
    Ni = 0 # 节点重要性
    label = -1 # 标签
    diffusion_flag = 0

    def __init__(self, index) -> None:
        self.index = index
    def __iter__(self):
        yield from {
            'index': self.index,
            'neighbors': self.neighbors,
            'degree': self.degree,
            'max_similarity_neighbor': self.max_similarity_neighbor,
            'max_similarity': self.max_similarity,
            'max_degree_neighbor': self.max_degree_neighbor,
            'max_neighbor': self.max_neighbor,
            'Ni': self.Ni,
            'label': self.label,
            'diffusion_flag': self.diffusion_flag
        }.items()
    def __str__(self) -> str:
        return json.dumps(dict(self), ensure_ascii=False)
    def __repr__(self) -> str:
        return self.__str__()

# ---------------------------------- Load Dataset -------------------------------------------
dataset_name = 'football' # name of dataset
path = './datasets/' + dataset_name + '.txt' # path to dataset
iteration = 1           # number of iterations for label selection step (mostly is set to 1 or 2)
merge_flag = 1         # merge_flag=0 -> do not merge //  merge_flag=1 -> do merge
write_flag = 0        # 1 means write nodes labels to file. 0 means do not write
modularity_flag = 0  # 1 means calculate modularity. 0 means do not calculate modularity
NMI_flag = 1        # 1 means calculate NMI. 0 means do not calculate NMI
# ------------------------- compute nodes neighbors and nodes degree --------------------------

graph = [] # 图，由Node组成
graph_index = {} # 记录每个序号的位置

# ------------------------- compute nodes neighbors and nodes degree --------------------------
# 读取数据库
# 获取邻接节点列表和每个节点的度
index = 0
with open(path) as f:
    for line in f:
        # 数据库结构
        # 每一行 = 一个节点的所有邻接节点序号，用'\t'隔开
        node = Node(index)
        if line.strip():
            # 有数据
            row = str(line.strip()).split('\t')
            neighbors = [int(i) for i in row ]
            node.neighbors = neighbors # 邻接节点列表
            node.degree = len(neighbors) # 当前节点的度
        else:
            # 没有数据
            node.neighbors = [] # 邻接节点列表
            node.degree = 0 # 当前节点的度
        graph.append(node) # 加入图
        graph_index[node.index] = index
        index += 1

N = index # 总节点数
start_time = time.time()

# ---------------------------------- Compute node importance -----------------------------------
# 计算节点和邻接节点相似性和节点重要性
for node in graph:
    if node.degree > 1:
        # 度 > 1
        Ni = 0 # 当前节点的重要性
        similaritys = {}
        degrees = {}
        for neighbor_index in node.neighbors:
            neighbor = graph[graph_index[neighbor_index]]
            # 每个邻接节点计算相似度
            intersect_size = len(list(set(node.neighbors) & set(neighbor.neighbors))) # 交集
            union_size = node.degree + neighbor.degree - intersect_size # 并集
            if node.degree >= neighbor.degree:
                difference = len(set(neighbor.neighbors).difference(set(node.neighbors))) # 差集
            else:
                difference = len(set(node.neighbors).difference(set(neighbor.neighbors)))# 差集
            # 相似性论文公式(1)
            similarity = (intersect_size / (intersect_size + union_size)) * (intersect_size / (1 + difference))
            similaritys[neighbor.index] = similarity
            # 重要性论文公式(2)
            Ni += similarity
            # 度
            degrees[neighbor.index] = neighbor.degree

        node.max_similarity_neighbor = max(similaritys, key=similaritys.get) # 最相似邻接节点
        node.max_similarity = similaritys[node.max_similarity_neighbor] # 最大相似度
        node.max_degree_neighbor = max(degrees, key=degrees.get) # 度最大的邻接节点
        node.Ni = Ni # 重要性
    elif node.degree == 1:
        # 度 = 1
        node.max_similarity_neighbor = -1 # 最相似邻接节点
        node.max_similarity = 0 # 最大相似度
        node.max_degree_neighbor = -1 # 度最大的邻接节点
        node.Ni = 0
    else:
        # 孤点
        node.max_similarity_neighbor = -1 # 最相似邻接节点
        node.max_similarity = -1 # 最大相似度
        node.max_degree_neighbor = -1 # 度最大的邻接节点
        node.Ni = 0
        
# 按重要性Ni排序
graph = sorted(graph, key=lambda n: n.Ni, reverse=True)

# 重新记录每个序号的位置
for index in range(N):
    node = graph[index]
    graph_index[node.index] = index

# ------------------------------ Select most similar neighbor -------------------------------
# 计算最大的邻接节点，并初始化label
for node in graph:
    if node.degree > 1:
        # 度 > 1
        if node.max_similarity == 0:
            # 最大相似度等于0
            # 选择邻接节点中度最大的
            node.max_neighbor = node.max_degree_neighbor  # 度最大的
            node.label = node.max_neighbor
        
        else:
            # 有最相似的
            node.max_neighbor = node.max_similarity_neighbor  # 最相似的
            max_similarity_neighbor = graph[graph_index[node.max_similarity_neighbor]]
            if node.Ni > max_similarity_neighbor.Ni:
                # 当前节点比最相似的邻接节点更重要
                node.label = node.index # 自己
            elif node.Ni == max_similarity_neighbor.Ni:
                node.label = min(node.index, node.max_neighbor) # 取小的那个
            else:
                node.label = node.max_neighbor # 最相似的
    else:
        # 度 <= 1
        node.max_neighbor = node.index # 自己
        node.label = node.index # 自己

# 选择与最大邻接节点相同的label
for node in graph:
    same_label = graph[graph_index[node.label]]
    node.label = same_label.label
        
# ----------------------------- Top 5 percent important nodes -------------------------------
# 前5%重要性的label三角扩散
top_5percent = ceil(N * 5 / 100) # 计算前5%有多少节点
top_5percent_nodes = graph[:top_5percent] # 得到前5%的节点

# 让最重要的节点、他最相似的节点，使用同样的标签
for node in top_5percent_nodes:
    temp_label = -1
    max_neighbor = graph[graph_index[node.max_neighbor]]
    if node.Ni >= max_neighbor.Ni:
        # 如果当前节点的重要性大于邻接节点
        # 邻接节点使用当前节点的标签
        temp_label = node.label
        max_neighbor.label = temp_label
    else:
        temp_label = max_neighbor.label
        node.label = temp_label

    # 标记已扩散
    node.diffusion_flag = 1
    max_neighbor.diffusion_flag = 1

    # 他两的共同邻居也使用同样的标签
    intersect = list(set(node.neighbors) & set(max_neighbor.neighbors)) # 交集
    for node_intersect_index in intersect:
        node_intersect = graph[graph_index[node_intersect_index]]
        node_intersect.label = temp_label
        node_intersect.diffusion_flag = 1 # 标记已扩散

# -------------------------------- Balanced Label diffusion ---------------------------------
# 取重要度最大和最小的两个点
high = 0
low = N -1
while high < low:
    # 对于大的点
    node_high = graph[high]
    if node_high.diffusion_flag == 0 and node_high.degree > 1:
        # 上一步未扩散且度>1
        # 所有邻接节点按label计算c_importance
        c_importance_high = {}

        for neighbor_index in node_high.neighbors:
            neighbor = graph[graph_index[neighbor_index]]
            # 论文公式(3)
            if c_importance_high.__contains__(neighbor.label):
                c_importance_high[neighbor.label] += neighbor.Ni
            else:
                c_importance_high[neighbor.label] = neighbor.Ni

        # 取邻接节点中c_importance最大的label
        node_high.label = max(c_importance_high, key = c_importance_high.get)
    
    # 对于小的点
    node_low = graph[low]
    if node_low.diffusion_flag == 0 and node_low.degree > 1:
        # 上一步未扩散且度>1
        # 所有邻接节点按label计算c_importance
        c_importance_low = {}

        for neighbor_index in node_low.neighbors:
            neighbor = graph[graph_index[neighbor_index]]
            # 论文公式(4)
            if c_importance_low.__contains__(neighbor.label):
                c_importance_low[neighbor.label] += node_low.degree * neighbor.degree
            else:
                c_importance_low[neighbor.label] = node_low.degree * neighbor.degree

        # 取邻接节点中c_importance最大的label
        node_low.label = max(c_importance_low, key = c_importance_low.get)

    high += 1
    low -= 1


# -------------------------- Give labels to nodes with degree=1 -----------------------------
# 处理度=1的点
for node in graph:
    if node.degree == 1:
        # 取邻居的label
        node.label = graph[graph_index[node.neighbors[0]]].label

# ---------------- Label selection step (the iterative part of algorithm) -------------------
# 标签选择，迭代次数由最开始参数决定
for iter in range(1):
    # for node in graph:
    # 这里是按节点顺序遍历，不是按重要性，不确定是否有影响
    for node_index in range(N):
        node = graph[graph_index[node_index]]
        if node.degree > 1:
            # 度>1
            # 统计邻居label出现的频率
            label_frequency = {}
            # 统计邻居社区的影响力
            effectiveness = {}
            for neighbor_index in node.neighbors:
                neighbor = graph[graph_index[neighbor_index]]
                if label_frequency.__contains__(neighbor.label):
                    label_frequency[neighbor.label] += 1
                else:
                    label_frequency[neighbor.label] = 1

                # 论文公式(5)
                if effectiveness.__contains__(neighbor.label):
                    effectiveness[neighbor.label] *= neighbor.Ni
                else:
                    effectiveness[neighbor.label] = neighbor.Ni

                # # 修改为参考论文2
                # # 找到他们的共同邻居，三角形
                # intersect = list(set(node.neighbors) & set(neighbor.neighbors)) # 交集
                # # 分别计算这些三角形的稳定性
                # stability = 0
                # for node_triangle_index in intersect:
                #     node_triangle = graph[graph_index[node_triangle_index]]

                #     if node.label == node_triangle.label:
                #         # 参考论文2的公式(2)
                #         intersect_size = len(list(set(node.neighbors) & set(neighbor.neighbors) & set(node_triangle.neighbors))) # 交集
                #         union_size = len(list(set(node.neighbors) | set(neighbor.neighbors) | set(node_triangle.neighbors))) # 并集
                #         stability += intersect_size / union_size
                    
                # if effectiveness.__contains__(neighbor.label):
                #     effectiveness[neighbor.label] += stability
                # else:
                #     effectiveness[neighbor.label] = stability

            # 最大频率
            max_frequency = max(label_frequency.values())
            # 取频率最大的label
            max_frequency_label = [k for k, v in label_frequency.items() if v == max_frequency]

            if len(max_frequency_label) == 1:
                # 如果只有一个最大频率的标签
                node.label = max_frequency_label[0]
            else:
                # 如果有多个，选择他们中社区影响度大的
                max_effectiveness = -1
                max_effectiveness_label = -1
                for label in max_frequency_label:
                    # # 如果有一个跟之前一样，直接不修改
                    # if node.label == label:
                    #     max_effectiveness_label = label
                    #     break
                    # 找到最大的
                    if max_effectiveness < effectiveness[label]:
                        max_effectiveness = effectiveness[label]
                        max_effectiveness_label = label
                node.label = max_effectiveness_label


# ---------------------------- Merge Small communities --------------------------------------
# 社区合并
if merge_flag == 1:
    # 统计每个社区有哪些节点
    community = {}
    for node in graph:
        if community.__contains__(node.label):
            community[node.label].append(node.index)
        else:
            community[node.label] = []
            community[node.label].append(node.index)

    # 最大的社区
    max_community_label = max(community, key = community.get)
    max_community = len(community[max_community_label])

    # 去除最大社区后平均社区大小
    avg_community = (N - max_community) / (len(community) - 1)

    # 选择小于平局大小的社区
    less_than_avg_community = {k : v for k, v in community.items() if len(v) < avg_community}

    # 按大小从小到大排序
    less_than_avg_community = {k : v for k, v in sorted(less_than_avg_community.items(), key=lambda item: len(item[1]), reverse=False)}

    if len(less_than_avg_community) > 0:
        for label in less_than_avg_community.keys():
            # 对每一个小于平均大小的社区
            # 计算每个节点的得分
            max_RS_in_community = -1
            max_RS_node_in_community = -1
            for node_index in less_than_avg_community[label]:
                node = graph[graph_index[node_index]]
                # 论文公式(6)
                RS = node.degree + node.Ni
                if RS > max_RS_in_community:
                    # RS最大的作为社区代表
                    max_RS_in_community = RS
                    max_RS_node_in_community = node.index

            # RS最大的作为社区代表
            represent_community = graph[graph_index[max_RS_node_in_community]]

            # # 社区代表最相似的邻居
            # represent_community_neighbor = graph[graph_index[represent_community.max_neighbor]]

            # new_label = -1
            # if represent_community.label != represent_community_neighbor.label:
            #     new_label = represent_community_neighbor.label
            # else:
            #     # 他们共同的邻居
            #     # 找到他们的共同邻居，三角形
            #     intersect = list(set(node.neighbors) & set(neighbor.neighbors)) # 交集
            
            # 社区代表的邻接节点，再算RS
            max_RS_in_neighbor = -1
            max_RS_node_in_neighbor = -1
            for neighbor_index in represent_community.neighbors:
                neighbor = graph[graph_index[neighbor_index]]
                # 论文公式(6)
                RS = neighbor.degree + neighbor.Ni
                if RS > max_RS_in_neighbor:
                    # RS最大的作为社区代表
                    max_RS_in_neighbor = RS
                    max_RS_node_in_neighbor = neighbor.index

            represent_neighbor = graph[graph_index[max_RS_node_in_neighbor]]

            # 判断是否需要更新社区的label
            if represent_community.label != represent_neighbor.label and max_RS_in_community < max_RS_in_neighbor:
                # 社区代表节点的label≠他RS最大的邻接节点的label
                # 社区代表节点的RS≤他RS最大的邻接节点
                for node_index in less_than_avg_community[label]:
                    # 这个社区里所有节点的label改为社区代表节点RS最大的邻接节点的label
                    node = graph[graph_index[node_index]]
                    node.label = represent_neighbor.label

# -------------------------- Total Time of Algorithm --------------------------------------
print(f'--- Total Execution time { round(time.time() - start_time, 6) } seconds ---')
# -------------------------------- Write to Disk ------------------------------------------
graph_orderby_index = sorted(graph, key=lambda n: n.index, reverse=False)
if write_flag == 1:

    if not os.path.exists('./results/'):
        os.makedirs('./results/')

    with open('./results/' + dataset_name + '.txt', 'w') as filehandle:
        for node in graph_orderby_index:
            filehandle.write(f'{node.label}\n')
# ---------------------------- Number of communities --------------------------------------
# 统计分了多少类
labels = {}
for node in graph_orderby_index:
    if labels.__contains__(node.label):
        labels[node.label] += 1
    else:
        labels[node.label] = 1
print(f'Number of Communities: {len(labels)}')
# ---------------------------------- Modularity -------------------------------------------
if modularity_flag ==1:
    t = 0
    for node in graph_orderby_index:
        t = t + node.degree
    edges = t / 2
    modu = 0
    are_neighbor = []

    for i in range(N):
        for j in range(N):
            if graph_orderby_index[i].label == graph_orderby_index[j].label:
                if graph_orderby_index[i].degree >= 1:
                    if j in graph_orderby_index[i].neighbors:
                        are_neighbor = 1
                    else:
                        are_neighbor = 0
                    modu = modu + (are_neighbor - ((graph_orderby_index[i].degree * graph_orderby_index[j].degree) / (2 * edges)))

    modularity_final = modu / (2 * edges)
    print(f'Modularity:  {modularity_final}')
# ------------------------------------- NMI -----------------------------------------------
if NMI_flag ==1:
    real_labels= loadtxt("./groundtruth/"+dataset_name+"_real_labels.txt", comments="#", delimiter="\t", unpack=False)
    detected_labels = []
    if dataset_name in ('karate','dolphins','polbooks','football'):
   
        for node in graph_orderby_index:
            detected_labels.append(node.label)
    
        detected_labels=np.array(detected_labels)
        print('NMI:  {}'.format(normalized_mutual_info_score(real_labels,detected_labels)))
    
    else:
        nodes_map = loadtxt("./datasets/nodes_map/" + dataset_name + "_nodes_map.txt", comments="#", delimiter="\t", unpack=False, dtype=int)

        for i in nodes_map:
            detected_labels.append(graph_orderby_index[i].label)
    
        print('NMI:  {}'.format(normalized_mutual_info_score(real_labels,detected_labels)))