import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from skimage import img_as_ubyte, color
import cv2

def minimum_cut(graph, source, target):
    # 创建一个副本图，用于记录剩余容量
    residual_graph = nx.MultiDiGraph()

    # 初始化最小割的值为0
    min_cut_value = 0

    # 初始化可达节点集合
    reachable_nodes = set()

    # 添加源点和汇点到副本图中
    residual_graph.add_node(source)
    residual_graph.add_node(target)

    # 将原图中的边属性添加到副本图中
    for u, v, capacity in graph.edges(data='capacity'):
        residual_graph.add_edge(u, v, capacity=capacity)
        residual_graph.add_edge(v, u, capacity=0)  # 添加反向边


    # 使用广度优先搜索找到从源点到汇点的一条增广路径
    def bfs(node):
        queue = [node]
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                for neighbor, attrs in residual_graph[node].items():
                    for attr in attrs.values():
                        if neighbor not in visited and attr['capacity'] > 0:
                            queue.append(neighbor)

    # 不断进行广度优先搜索，直到无法找到增广路径为止
    while True:
        # 初始化已访问节点集合
        visited = set()

        # 使用广度优先搜索找到一条增广路径
        bfs(source)

        # 如果汇点在已访问节点集合中，则存在增广路径
        if target in visited:
            # 找到增广路径上剩余容量最小的边
            min_residual_capacity = min(residual_graph[u][v][0]['capacity'] for u, v in nx.edge_dfs(residual_graph, source=source, orientation='reverse') if residual_graph.has_edge(u, v))

            # 遍历增广路径上的边，更新剩余容量
            for u, v in nx.edge_dfs(graph, source=source, orientation='reverse'):
                # 更新剩余容量
                if residual_graph.has_edge(u, v):
                    residual_graph[u][v]['capacity'] -= min_residual_capacity
                    if residual_graph[u][v]['capacity'] == 0:
                        residual_graph.remove_edge(u, v)
                if residual_graph.has_edge(v, u):
                    residual_graph[v][u]['capacity'] += min_residual_capacity
                    if residual_graph[v][u]['capacity'] == 0:
                        residual_graph.remove_edge(v, u)
            # 增加最小割的值
            min_cut_value += min_residual_capacity

            # 将可达节点添加到可达节点集合中
            reachable_nodes.update(visited)
        else:
            # 如果无法找到增广路径，则退出循环
            break

    # 不可达节点集合为所有节点减去可达节点集合
    non_reachable_nodes = set(graph.nodes) - reachable_nodes

    # 返回最小割的值和可达节点集合、不可达节点集合
    return min_cut_value, (reachable_nodes, non_reachable_nodes)

def graph_cut_segmentation(image):
    gray_image = img_as_ubyte(color.rgb2gray(image))
    g = nx.DiGraph()
    height, width = gray_image.shape
    nodes = np.arange(height * width).reshape((height, width))
    g.add_node('s')
    g.add_node('t')
    for i in range(height):
        for j in range(width):
            node_id = nodes[i, j]
            g.add_node(node_id)
            g.add_edge('s', node_id, capacity=gray_image[i, j])
            g.add_edge(node_id, 't', capacity=255 - gray_image[i, j])
            if i > 0:
                g.add_edge(node_id, nodes[i-1, j], capacity=1)
            if i < height - 1:
                g.add_edge(node_id, nodes[i+1, j], capacity=1)
            if j > 0:
                g.add_edge(node_id, nodes[i, j-1], capacity=1)
            if j < width - 1:
                g.add_edge(node_id, nodes[i, j+1], capacity=1)
    cut_value, partition = minimum_cut(g, 's', 't')
    reachable, non_reachable = partition
    segmented = np.zeros_like(gray_image)
    for node_id in reachable:
        if node_id != 's':
            i, j = np.unravel_index(node_id, (height, width))
            segmented[i, j] = 255
    return segmented

images = ['./bird.png', './flower.png', './pokemon.png']
for image_path in images:
    image = cv2.imread(image_path)
    segmented = graph_cut_segmentation(image)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(segmented, cmap='gray')
    ax[1].set_title('Segmentation Result')
    ax[1].axis('off')
    plt.show()
