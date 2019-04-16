---
layout: post
title: np.meshgrid
---


### 生成网格点坐标

plt.plot() 可以接受矩阵作为坐标信息，两个矩阵维度相同，根据对应关系来生成点。此时，matplotlib会将提供横坐标矩阵中的每一列对应的点当作同一条线。


'''
 plt.plot(x,y,
          marker='.',#点的形状
          markersize=10, #点的大小
          linestyle='-.') #线形为点划线
'''

