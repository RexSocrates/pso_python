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
    def __init__(self, pN, dim, max_iter, boundary):
        self.w = 0.8
        self.c1 = 2 # 學習因子
        self.c2 = 2
        self.r1 = 0.4417343281858497 # 0 ~ 1 之間的隨機值 保持群體多樣性
        self.r2 = 0.019908997513138127
        self.pN = pN  # 粒子數量
        self.dim = dim  # 搜索維度
        self.max_iter = max_iter  # 反覆運算次數
        self.boundary = boundary #  探索邊界
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置和速度
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))  # 個體經歷的最佳位置
        self.gbest = np.zeros((1, self.dim))  # 全域最佳位置
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
    def rosenbrockFunction(self,x):    
        total = 0    
        length = len(x)
        for i in range(length - 1):
            total += 100 * (x[i + 1] - x[i]**2)**2 + (x[i] - 1)**2
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
        for i in range(self.pN):
            for j in range(self.dim):
                # 給予每個粒子初始化的位置和速度
                self.X[i][j] = random.uniform(0, 1)
                self.V[i][j] = random.uniform(0, 1)
            self.pbest[i] = self.X[i]
            if option == 1 :
                tmp = self.sphereFunction(self.X[i])
            if option == 2 :
                tmp = self.rosenbrockFunction(self.X[i])
            if option == 3 :
                tmp = self.rastriginFunction(self.X[i])
            
            self.p_fit[i] = tmp
            if tmp < self.fit:
                self.fit = tmp
                self.gbest = self.X[i]
                
# ----------------------更新粒子位置----------------------------------

    def iterator(self):
        fitness = []
        for t in range(self.max_iter):
            for i in range(self.pN):  # 更新gbest\pbest
                if option == 1 :
                    temp = self.sphereFunction(self.X[i])
                if option == 2 :
                    temp = self.rosenbrockFunction(self.X[i])
                if option == 3 :
                    temp = self.rastriginFunction(self.X[i])
                
                # 更新個體最優
                if temp < self.p_fit[i]:
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    if self.p_fit[i] < self.fit:  # 更新全域最優
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]
            for i in range(self.pN):
                # 更新所有粒子的位置與速度
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
                            self.c2 * self.r2 * (self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
                
                
                # 如果更新後超出邊界則從另一頭出現
                for j in range(len(self.X[i])) :
                    if self.X[i][j] > self.boundary[0] :
                        self.X[i][j] = self.boundary[-1]
                    
                    if self.X[i][j] < self.boundary[-1] :
                        self.X[i][j] = self.boundary[0]
                
            # 紀錄目前群體最佳解
            fitness.append(self.fit)
            print(self.X[0], end=" ")
            print("目前群體最佳:", self.fit, " 目前代數:", t+1)  # 輸出最優值
        return fitness
    
    def iterator1(self):
        fitness = []
        for t in range(self.max_iter):
            for i in range(self.pN):  # 更新gbest\pbest
                if option == 1 :
                    temp = self.sphereFunction(self.X[i])
                temp = self.rastriginFunction(self.X[i]) #引值  第二題 題目  #取return 值 total
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
                
                for j in range(len(self.X[i])):
                    #如果大於邊界讓他回來
                    if self.X[i][j]>self.boundary[0]:
                       self.X[i][j]=self.boundary[1]
                       
                    if self.X[i][j]<self.boundary[1]:   
                       self.X[i][j]=self.boundary[0]
                
                
            fitness.append(self.fit)
            print(self.X[0], end=" ")
            print(self.fit , '目前代數 : ', t+1 ,'目前的r1.r2是:',self.r1,'與',self.r2)  # 輸出最優值

                
        return fitness

# ----------------------程式執行-----------------------

numberOfParticles = int(input("輸入粒子數量(100):"))
# dim = int(input("輸入維度(2):"))
dim = 2
max_iter = int(input("最大迭代數(1000):"))

print("1. Sphere")
print("2. Rosenbrock")
print("3. Rastrigin")
option = int(input("選擇函數:"))


# 探索邊界
# sphere function
if option == 1 :
    boundary = [100, -100]

# rosenbrock function
if option == 2 :
    boundary = [50, -50]

# rastrigin function
if option == 3 :
    boundary = [5.12, -5.12]


my_pso = PSO(pN=numberOfParticles, dim=dim, max_iter=max_iter, boundary = boundary)
my_pso.init_Population()
fitness = my_pso.iterator1()

# -------------------畫圖--------------------
plt.figure(1)
plt.title("Particles :" + str(numberOfParticles) + " Dimension:" + str(dim) + " Iteration:" + str(max_iter))
plt.xlabel("iterators", size=14)
plt.ylabel("fitness", size=14)
t = np.array([t for t in range(0, max_iter)])
fitness = np.array(fitness)
plt.plot(t, fitness, color='b', linewidth=3)
plt.show()
