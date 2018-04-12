#  coding: utf-8 
"""
版權聲明：© 2018 • OmegaXYZ-版權所有 轉載請注明出處 - QQ：644327005
http://blog.csdn.net/xyisv/article/details/79058574

注意pN是指初始種群，一般來說初始種群越大效果越好
dim是優化的函數維度，常見的初等函數和初等複合函數都是1維
max_iter是反覆運算次數
本demo-1的優化函數是x^2-4x+3，顯然這個函數在x=2時取最小值-1
本demo-2的優化函數(目標函數)是Sphere函數 f(x)=Σx^2，這個函數最佳解是0
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import math

# ----------------------PSO參數設置---------------------------------
class PSO():
    def __init__(self, pN, dim, max_iter):
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.6
        self.r2 = 0.3
        self.pN = pN  # 粒子數量
        self.dim = dim  # 搜索維度
        self.max_iter = max_iter  # 反覆運算次數
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置和速度
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))  # 個體經歷的最佳位置和全域最佳位置
        self.gbest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)  # 每個個體的歷史最佳適應值
        self.fit = 1e10  # 全域最佳適應值

# ---------------------目標函數 Sphere函數-----------------------------
    """
    def function(self, X):
        return X**2-4*X+3
    """
   
    def sphereFunction(self,x):    
        total = 0    
        length = len(x)    
        x = x**2    
        for i in range(length):    
            total += x[i]    
        return total      
# ---------------------目標函數 Rosenbrock函數-----------------------------
    def rosenbrockFunction(self,x1, x2):    
        total = 0    
        length = len(x1)
        for i in range(length):
            total += 100 * (x2[i] - x1[i]**2)**2 + (x1[i] - 1)**2
        return total
# ---------------------目標函數 Rastrigin函數-----------------------------
    def rastriginFunction(self,x):    
        total = 0    
        length = len(x)    
        for i in range(length):    
            total += x[i]**2 - 10 * math.cos(2 * math.pi * x[i]) + 10
        return total
# ---------------------初始化種群----------------------------------
    def init_Population(self):
        for i in range(self.pN - 1):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(0, 1)
                self.V[i][j] = random.uniform(0, 1)
            self.pbest[i] = self.X[i]
            tmp = self.rosenbrockFunction(self.X[i], self.X[i+1])
            self.p_fit[i] = tmp
            if tmp < self.fit:
                self.fit = tmp
                self.gbest = self.X[i]
                
# ----------------------更新粒子位置----------------------------------
    def iterator(self):
        fitness = []
        for t in range(self.max_iter):
            for i in range(self.pN - 1):  # 更新gbest\pbest
                temp = self.rosenbrockFunction(self.X[i], self.X[i+1])
                if temp < self.p_fit[i]:  # 更新個體最優
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    if self.p_fit[i] < self.fit:  # 更新全域最優
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]
            for i in range(self.pN):
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
                            self.c2 * self.r2 * (self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
            fitness.append(self.fit)
            print(self.X[0], end=" ")
            print(self.fit)  # 輸出最優值
        return fitness

# ----------------------程式執行-----------------------
my_pso = PSO(pN=10, dim=2, max_iter=100)
my_pso.init_Population()
fitness = my_pso.iterator()

# -------------------畫圖--------------------
plt.figure(1)
plt.title("figure1")
plt.xlabel("iterators", size=14)
plt.ylabel("fitness", size=14)
t = np.array([t for t in range(0, 100)])
fitness = np.array(fitness)
plt.plot(t, fitness, color='b', linewidth=3)
plt.show()
