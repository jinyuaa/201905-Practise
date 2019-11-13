# sklearn 库中函数实现
# make_blobs 聚类生成器

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 实现逻辑：
# 1、随机生成 K 个初始点作为质心
# 2、将数据集中的数据按距离质心的远近分到各个簇中
# 3、将各个簇中的数据求平均值，作为新的质心，重复上一步，直至所有簇不再改变

# 随机生成400条数据、4类数据、方差一致为0.5、随机种子
x, y_label = make_blobs(n_samples=400, centers=4, cluster_std=0.5, random_state=0)
print(x.shape)
plt.scatter(x[:, 0], x[:, 1], s=50, marker='x')
plt.grid()
plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(x)
y_means = kmeans.predict(x)

# c-color点的颜色('b');cmap,Colormap实例(None);s点的大小(默认20);
# alpha点的亮度(None);marker点的形状('o');label点的标签
plt.scatter(x[:, 0], x[:, 1], c=y_means, cmap='Dark2', s=50, alpha=0.4, marker='x')

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c=[0, 1, 2, 3], cmap='Dark2', s=70, marker='o')
plt.title("k-means 400 points")
plt.xlabel("Value1")
plt.ylabel("Value2")
plt.grid()
plt.show()
