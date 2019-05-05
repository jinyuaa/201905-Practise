from numpy_CNN import NumpyCNN as npcnn
from skimage import data, color
import numpy as np
import matplotlib.pyplot as plt
'''
使用NumPy实现卷积神经网络。
只创建了三个层，分别是卷积、ReLU和最大池。所涉及的主要步骤如下:
1、读取输入图像
2、准备过滤
3、Conv层:将每个滤波器与输入图像进行卷积。
4、ReLU层:对feature map (conv层输出)应用ReLU激活函数。
5、最大池化层:对ReLU层的输出应用池化操作。
6、堆叠conv、ReLU和最大池层
'''

img = data.chelsea()
img = color.rgb2grey(img)

# first conv layer
print('第一层----------------------------------------')
layer1_filter = np.zeros((2, 3, 3))
print(type(layer1_filter))
layer1_filter[0, :, :] = np.array([[[-1, 0, 1],
                                    [-1, 0, 1],
                                    [-1, 0, 1]]])
layer1_filter[1, :, :] = np.array([[[1,   1,  1],
                                    [0,   0,  0],
                                    [-1, -1, -1]]])
layer1_feature_map = npcnn.conv(img, layer1_filter)
layer1_feature_map_relu = npcnn.relu(layer1_feature_map)
layer1_feature_map_relu_pool = npcnn.pooling(layer1_feature_map_relu, 2, 2)

# second conv layer
print('第二层----------------------------------------')
layer2_filter = np.random.rand(3, 5, 5, layer1_feature_map_relu_pool.shape[-1])
# print(layer2_filter)
layer2_feature_map = npcnn.conv(layer1_feature_map_relu_pool, layer2_filter)
layer2_feature_map_relu = npcnn.relu(layer2_feature_map)
layer2_feature_map_relu_pool = npcnn.pooling(layer2_feature_map_relu, 2, 2)

# third conv layer
print('第三层----------------------------------------')
layer3_filter = np.random.rand(1, 7, 7, layer2_feature_map_relu_pool.shape[-1])
# print(layer3_filter)
layer3_feature_map = npcnn.conv(layer2_feature_map_relu_pool, layer3_filter)
layer3_feature_map_relu = npcnn.relu(layer3_feature_map)
layer3_feature_map_relu_pool = npcnn.pooling(layer3_feature_map_relu, 2, 2)

# Graphing results
fig0, ax0 = plt.subplots(nrows=1, ncols=1)
ax0.imshow(img).set_cmap("gray")
ax0.set_title("Input Image")
# 显示坐标刻度
# ax0.get_xaxis().set_ticks([])
# ax0.get_yaxis().set_ticks([])
# (bounding box)bbox_inches = 'tight' 可以去除坐标轴占用的空间, 只保存图形的给定部分，解决图片不清晰、不完整的问题
plt.savefig("in_img.png", bbox_inches="tight")
plt.close(fig0)

# Layer 1
fig1, ax1 = plt.subplots(nrows=3, ncols=2)
ax1[0, 0].imshow(layer1_feature_map[:, :, 0]).set_cmap("gray")
ax1[0, 0].get_xaxis().set_ticks([])
ax1[0, 0].get_yaxis().set_ticks([])
ax1[0, 0].set_title("L1-Map1")

ax1[0, 1].imshow(layer1_feature_map[:, :, 1]).set_cmap("gray")
ax1[0, 1].get_xaxis().set_ticks([])
ax1[0, 1].get_yaxis().set_ticks([])
ax1[0, 1].set_title("L1-Map2")

ax1[1, 0].imshow(layer1_feature_map_relu[:, :, 0]).set_cmap("gray")
ax1[1, 0].get_xaxis().set_ticks([])
ax1[1, 0].get_yaxis().set_ticks([])
ax1[1, 0].set_title("L1-Map1ReLU")

ax1[1, 1].imshow(layer1_feature_map_relu[:, :, 1]).set_cmap("gray")
ax1[1, 1].get_xaxis().set_ticks([])
ax1[1, 1].get_yaxis().set_ticks([])
ax1[1, 1].set_title("L1-Map2ReLU")

ax1[2, 0].imshow(layer1_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
ax1[2, 0].get_xaxis().set_ticks([])
ax1[2, 0].get_yaxis().set_ticks([])
ax1[2, 0].set_title("L1-Map1ReLUPool")

ax1[2, 1].imshow(layer1_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
ax1[2, 1].get_xaxis().set_ticks([])
ax1[2, 1].get_yaxis().set_ticks([])
ax1[2, 1].set_title("L1-Map2ReLUPool")

plt.savefig("L1.png", bbox_inches="tight")
plt.close(fig1)

# Layer 2
fig2, ax2 = plt.subplots(nrows=3, ncols=3)
ax2[0, 0].imshow(layer2_feature_map[:, :, 0]).set_cmap("gray")
ax2[0, 0].get_xaxis().set_ticks([])
ax2[0, 0].get_yaxis().set_ticks([])
ax2[0, 0].set_title("L2-Map1")

ax2[0, 1].imshow(layer2_feature_map[:, :, 1]).set_cmap("gray")
ax2[0, 1].get_xaxis().set_ticks([])
ax2[0, 1].get_yaxis().set_ticks([])
ax2[0, 1].set_title("L2-Map2")

ax2[0, 2].imshow(layer2_feature_map[:, :, 2]).set_cmap("gray")
ax2[0, 2].get_xaxis().set_ticks([])
ax2[0, 2].get_yaxis().set_ticks([])
ax2[0, 2].set_title("L2-Map3")

ax2[1, 0].imshow(layer2_feature_map_relu[:, :, 0]).set_cmap("gray")
ax2[1, 0].get_xaxis().set_ticks([])
ax2[1, 0].get_yaxis().set_ticks([])
ax2[1, 0].set_title("L2-Map1ReLU")

ax2[1, 1].imshow(layer2_feature_map_relu[:, :, 1]).set_cmap("gray")
ax2[1, 1].get_xaxis().set_ticks([])
ax2[1, 1].get_yaxis().set_ticks([])
ax2[1, 1].set_title("L2-Map2ReLU")

ax2[1, 2].imshow(layer2_feature_map_relu[:, :, 2]).set_cmap("gray")
ax2[1, 2].get_xaxis().set_ticks([])
ax2[1, 2].get_yaxis().set_ticks([])
ax2[1, 2].set_title("L2-Map3ReLU")

ax2[2, 0].imshow(layer2_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
ax2[2, 0].get_xaxis().set_ticks([])
ax2[2, 0].get_yaxis().set_ticks([])
ax2[2, 0].set_title("L2-Map1ReLUPool")

ax2[2, 1].imshow(layer2_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
ax2[2, 1].get_xaxis().set_ticks([])
ax2[2, 1].get_yaxis().set_ticks([])
ax2[2, 1].set_title("L2-Map2ReLUPool")

ax2[2, 2].imshow(layer2_feature_map_relu_pool[:, :, 2]).set_cmap("gray")
ax2[2, 2].get_xaxis().set_ticks([])
ax2[2, 2].get_yaxis().set_ticks([])
ax2[2, 2].set_title("L2-Map3ReLUPool")

plt.savefig("L2.png", bbox_inches="tight")
plt.close(fig2)

# Layer 3
fig3, ax3 = plt.subplots(nrows=1, ncols=3)
ax3[0].imshow(layer3_feature_map[:, :, 0]).set_cmap("gray")
ax3[0].get_xaxis().set_ticks([])
ax3[0].get_yaxis().set_ticks([])
ax3[0].set_title("L3-Map1")

ax3[1].imshow(layer3_feature_map_relu[:, :, 0]).set_cmap("gray")
ax3[1].get_xaxis().set_ticks([])
ax3[1].get_yaxis().set_ticks([])
ax3[1].set_title("L3-Map1ReLU")

ax3[2].imshow(layer3_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
ax3[2].get_xaxis().set_ticks([])
ax3[2].get_yaxis().set_ticks([])
ax3[2].set_title("L3-Map1ReLUPool")

plt.savefig("L3.png", bbox_inches="tight")
plt.close(fig3)