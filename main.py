import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from skimage import data, img_as_ubyte, color
import cv2

def graph_cut_segmentation(image):
    # 将图像转换为灰度图像
    gray_image = img_as_ubyte(color.rgb2gray(image))

    # 创建图
    g = nx.DiGraph()

    # 添加图的节点
    height, width = gray_image.shape
    nodes = np.arange(height * width).reshape((height, width))

    # 添加源点和汇点
    g.add_node('s')
    g.add_node('t')

    # 添加图的边和容量
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

    # 使用最大流最小割算法进行图像分割
    cut_value, partition = nx.minimum_cut(g, 's', 't')
    reachable, non_reachable = partition

    # 获取分割结果
    segmented = np.zeros_like(gray_image)
    for node_id in reachable:
        if node_id != 's':
            i, j = np.unravel_index(node_id, (height, width))
            segmented[i, j] = 255

    return segmented

images = ['./bird.png', './flower.png', './pokemon.png']
for _image in images:
    # 读取图像
    image = cv2.imread(_image)

    # 进行图像分割
    segmented = graph_cut_segmentation(image)

    # 显示原始图像和分割结果
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(segmented, cmap='gray')
    ax[1].set_title('Segmentation Result')
    ax[1].axis('off')
    plt.show()
