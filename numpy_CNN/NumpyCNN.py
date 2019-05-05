# 仅使用 Numpy 完成 CNN 的搭建  对图片处理
# 灰度图 即舍弃 channel , 2 个 3X3 的卷积核得到 feature_maps
from skimage import data, io, color
import numpy as np
import logging as log

# log.basicConfig(level=log.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

# data.chelsea():chelsea是scikit-image.data模块下自带的图片，小猫图片
# skimage.io读出来的数据是numpy格式的，且是(height, width, channel)
# img = data.chelsea()
# io.imshow(img)
# io.show()
# print('-----------------------------')

'''
image = io.imread('D:/jinyu1.jpg')
io.imshow(image)
print("img的类型", type(image))
print("img的尺寸", image.shape)
print("img的宽度", image.shape[0])
print("img的高度", image.shape[1])
print("img的通道数", image.shape[2])
'''
# 对图像进行灰度处理
# img = color.rgb2gray(img)
# io.imshow(img)
# io.show()
# 变成灰度图后，舍弃了通道数，即变成二维图像矩阵
'''
print("img的类型", type(img))
print("img的尺寸", img.shape)
print("img的宽度", img.shape[0])
print("img的高度", img.shape[1])
# print("img的通道数", img.shape[2])
print("img的总像素个数", img.size)
print("img的最大像素值", img.max())
print("img的最小像素值", img.min())
print("img的平均像素值", img.mean())
'''
# ----------------------------------卷积层-------------------------------
# 第一个卷积层Conv准备滤波器(Layer_1)
# 创建两个2个3x3大小的滤波器
# 如果处理是彩色图，则滤波器大小为(3,3,3)，最后一个3表示深度，(2,3,3,3)
# Layer_1_filter = np.zeros((2, 3, 3))
# 第一个 3X3 滤波器 可以写成 print(Layer_1_filter[0, :, :])
# log.debug(Layer_1_filter[0])
# log.debug(img.shape)
# 卷积核的取值在没有以往的学习经验下，可由函数随机产生，再逐步训练调整
'''
Layer_1_filter[0, :, :] = np.array([[[-1, 0, 1], 
                                     [-1, 0, 1], 
                                     [-1, 0, 1]]])
Layer_1_filter[1, :, :] = np.array([[[1,   1,  1],
                                     [0,   0,  0],
                                     [-1, -1, -1]]])
'''


# 自己构造 conv 函数
# 只接受 img, conv_filter 两个参数 判断单个filter是否单通道
def conv(img, conv_filter):
    # 检查 滤波器的深度 设置 与 图像尺寸是否匹配
    # 首先检查图像与滤波器是否有深度通道
    '''
    if len(img.shape) > 2 or len(conv_filter.shape) > 3:
        # 若存在，检查其通道数是否相等(=3,rgb彩图)
        if img.shape[-1] != conv_filter[-1]:
            print("Error")
    # 检查滤波器尺寸是否相等 (每个滤波器的大小是相等的)
    if conv_filter.shape[1] != conv_filter.shape[2]:
        print("Error")
    # 检查滤波器是否为奇数 (且滤波器的大小应该是奇数)
    if conv_filter.shape[1] % 2 == 0:
        print("Error")
    '''
    # 用来保存图像和滤波器进行卷积后的结果
    # 默认步幅为1 无填充
    feature_maps = np.zeros((img.shape[0]-conv_filter.shape[1]+1,
                             img.shape[1]-conv_filter.shape[1]+1,
                             conv_filter.shape[0]))
    # 计算 第一个/第二个 filter与img卷积 conv_函数
    for filter_num in range(conv_filter.shape[0]):
        print("Filter:", filter_num + 1)
        curr_filter = conv_filter[filter_num, :, :]
        # 滤波器的大小(3x3)
        log.debug(len(curr_filter.shape))
        # 检查单个滤波器是否有多通道(rgb图)
        if len(curr_filter.shape) > 2:
            # 相当于sum0初值
            # 将多通道的每个filter与img卷积并对结果求和，返回 feature_maps
            conv_map = calculate_conv(img[:, :, 0], curr_filter[:, :, 0])
            for ch_num in range(1, curr_filter.shape[-1]):
                conv_map = conv_map + calculate_conv(img[:, :, ch_num], curr_filter[:, :, ch_num])
        # 单个滤波器单个通道 1个3x3
        else:
            conv_map = calculate_conv(img, curr_filter)

        feature_maps[:, :, filter_num] = conv_map
    return feature_maps


# 进行卷积操作
# img.shape : (300, 451)
def calculate_conv(img, conv_filter):
    filter_size = conv_filter.shape[0]  # 3
    result = np.zeros(img.shape)  # (300, 451)
    # 逐步卷积 (1:)(1:446)
    # np.uint16 无符号整数，0 至 2^16=65535
    # 1:300-1:298,0:297
    for r in np.arange(0, img.shape[0]-filter_size+1):
        for c in np.arange(0, img.shape[1]-filter_size+1):
            # 在 img 上以步长为 1 滑动取值
            curr_region = img[r:r+filter_size, c:c+filter_size]
            # 将当前区域与 filter 相乘
            curr_result = curr_region * conv_filter
            # 求和
            conv_sum = np.sum(curr_result)
            result[r, c] = conv_sum
    # final_result (298,449)
    final_result = result[0:result.shape[0]-filter_size+1, 0:result.shape[1]-filter_size+1]
    return final_result


# 构建好滤波器后，与输入图像进行卷积
# Layer_1_feature_map = conv(img, Layer_1_filter)
# (298, 449, 2)
# L1_map1 = Layer_1_feature_map[:, :, 0]
'''
print("----------------------------------------")
print("img的类型", type(L1_map1))
print("img的尺寸", L1_map1.shape)
print("img的宽度", L1_map1.shape[0])
print("img的高度", L1_map1.shape[1])
# print("img的通道数", L1_map1.shape[2])
print("img的总像素个数", L1_map1.size)
print("img的最大像素值", L1_map1.max())
print("img的最小像素值", L1_map1.min())
print("img的平均像素值", L1_map1.mean())
io.imshow(L1_map1)
io.show()
'''
# L1_Map2 = Layer_1_feature_map[:, :, 1]


# -------------------------------ReLU激活函数层----------------------------
# ReLU层激活函数应用于conv层输出的每个特征图上


def relu(feature_map):
    relu_out = np.zeros(feature_map.shape) # (298,449,2)
    for map_num in range(feature_map.shape[-1]):
        for r in np.arange(0, feature_map.shape[0]):
            for c in np.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = np.maximum(feature_map[r, c, map_num], 0)
    return relu_out


# Layer_1_feature_map_relu = relu(Layer_1_feature_map)
# log.debug(Layer_1_feature_map_relu.shape)
# L1_relu_map1 = Layer_1_feature_map_relu[:, :, 0]
# io.imshow(L1_relu_map1)
# io.show()
# print("img的最大像素值", L1_relu_map1.max())
# print("img的最小像素值", L1_relu_map1.min())

# -------------------------------最大池化层----------------------------
# 最大池化函数 max pooling


# feature_map (298 449 2)
def pooling(feature_map, size, stride):
    # pool_out (149, 224, 2)
    pool_out = np.zeros(((feature_map.shape[0] // stride),
                         (feature_map.shape[1] // stride),
                         feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in np.arange(0, feature_map.shape[0]-stride+1, stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[1]-stride+1, stride):
                pool_out[r2, c2, map_num] = np.max(feature_map[r:r+size, c:c+size])
                c2 = c2 + 1
            r2 = r2 + 1
    return pool_out

'''
Layer_1_feature_map_relu_pooling = pooling(Layer_1_feature_map_relu, 2, 2)
log.debug(Layer_1_feature_map_relu_pooling.shape)
log.debug(Layer_1_feature_map_relu_pooling.shape)
L1_relu_pooling_map1 = Layer_1_feature_map_relu_pooling[:, :, 0]
io.imshow(L1_relu_pooling_map1)
io.show()
print("img的最大像素值", L1_relu_pooling_map1.max())
print("img的最小像素值", L1_relu_pooling_map1.min())
'''






