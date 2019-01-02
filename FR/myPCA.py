#!/usr/bin/env python
# coding: utf-8

# 先看這張圖 https://img-blog.csdn.net/20171022165350939

# 由上一次程式碼可知，PCA 的步驟如下
# (1) 將數據結構化成多維度陣列
# (2) 計算矩陣均值
# (3) 零均值化，也就是陣列的所有數都減去平均
# (4) 轉軸
# (5) 計算共變異數矩陣
# (6) 計算特徵矩陣向量
# (7) 取出主要特徵

import numpy as np
from numpy import linalg as LA
import cv2
import os
import time

startTime = time.time()

print("="*100)

# ====================================================================================================
#    讀取所有圖片位置
# ====================================================================================================

PATH = 'C:\FR\Faces2' # 指定圖像資料夾路徑
facesDir = os.listdir(PATH)  # 開啟圖像資料夾
trainDataFiles = open('train.txt', 'w') # 以寫入模式開啟 train.txt
testDataFiles = open('test.txt', 'w') # 以寫入模式 test.txt

print("已讀取指定資料夾 %s，包含 %s 個檔案" %(PATH, len(facesDir)))

# Len = 取得內容的個數
# 迴圈是要取得 Faces 資料夾內的所有資料夾
for i in range(len(facesDir)):
    d2 = os.listdir (PATH + '/%s' %(facesDir[i])) # 進入 Faces 資料夾再下一層的各資料夾

    for j in range(len(d2)-1): # 注意這邊減去一張（最後一張不拿），是為了保留等等測試集要用的圖片
        str1 = PATH + '/%s/%s' %(facesDir[i], d2[j]) # 取得每個子資料夾的內容檔案路徑
        print("正在寫入第 %s 人的圖像檔案路徑至訓練集 Train.txt：%s" %(i + 1 , str1))
        trainDataFiles.write("%s %d\n" % (str1, i)) # 將每個內容檔案路徑都存放到 train.txt，並標註是第幾個人

    for h in range(1):
        str1 = PATH + '/%s/%s' %(facesDir[i], d2[-h]) # 取得每個資料夾中的最後一張圖片的路徑
        print("正在寫入第 %s 人的圖像檔案路徑至測試集  Test.txt：%s" %(i + 1, str1))
        testDataFiles.write("%s %d\n" % (str1, i)) # 將每個資料夾最後一張圖片的路徑都存放到 test.txt，並標註是第幾個人

# 關閉 train.txt 和 test.txt
trainDataFiles.close() 
testDataFiles.close()

# 上面這個區塊的目的在於：將圖片庫中的每個資料夾逐一打開，並將最後一張放到測試集的文字檔、以及將其他張照片放到訓練集的文字檔

print("已成功將資料寫入至 train.txt 和 test.txt","="*100)

# ====================================================================================================
#    讀取圖片並轉換成向量矩陣
# ====================================================================================================

# 定義讀取圖片的函式
def load_img(f):
    f = open(f) # 開啟傳入的路徑／檔案，這邊會用於 train.txt 和 test.txt
    lines = f.readlines() # 讀取所有的行並存成一個列表
    imgs, lab=[], [] 

    # 逐行操作
    for i in range(len(lines)):
        fn, label = lines[i].split(' ') # 取出路徑和標籤，用空格斷詞 Faces/9326871/9326871.9.jpg 0 就會被斷成 Faces/9326871/9326871.9.jpg 和 0

        print("正在處理圖像：%s" %fn)
        
        im1 = cv2.imread(fn) # 讀取圖檔
        im1 = cv2.resize(im1, (64,64)) # 調整圖片尺寸
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY) # 灰階

        vec = np.reshape(im1, [-1]) # -1 => 長長一條的矩陣
        print("圖像 %s 已轉換為一維矩陣：%s" %(fn, vec))
        # reshape 用於轉換成指定模樣的矩陣
        # 請參閱 https://blog.csdn.net/u014722627/article/details/55259617
        # 請參閱 https://blog.csdn.net/DeniuHe/article/details/77377237 

        imgs.append(vec) # 在 img 列表加入轉換好的圖片矩陣
        lab.append(int(label)) # 在 lab 列表裡面加入標籤
        
    # 將兩個列表結構化成 ndarray 多維度陣列
    imgs = np.asarray(imgs, np.float32)
    lab = np.asarray(lab, np.int32)
    return imgs, lab 


x, y = load_img('train.txt')
tx, ty = load_img('test.txt')

print("="*100,"\n處理完成\n訓練集已轉換為 %s 的圖像矩陣和 %s 的標籤\n測試集已轉換為 %s 的圖像矩陣和 %s 的標籤" %(x.shape, y.shape, tx.shape, ty.shape))
# (2147,4096) & (113,4096)
print("="*100)

# ====================================================================================================
#    PCA
# ====================================================================================================

m1 = np.mean(x, 0) # 計算 訓練集 矩陣均值
print("訓練集 矩陣均值：", m1.shape)

xm = x - m1 # 零均值化，也就是陣列的所有數都減去平均
print("訓練集 零均值化：", xm.shape)

# xm /= np.std(xm, axis = 0)
# 這邊如果對原始資料做歸一化，成功率會砍半，因為原始資料本來就很像
# 但是若對特徵做整理效果反而會提升

C = np.cov(xm) # 計算 共變異數矩陣 # rowvar = 0
print("訓練集 共變異數：", C.shape) 


print("\n正在計算特徵向量")

w, v = LA.eig(C) # 計算 特徵矩陣向量
print("\n訓練集 特徵矩陣向量\n特徵值：", w.shape)
print("特徵向量：", v.shape)

v = -np.sort(-v, axis = 1)

p1 = v[:, 0:3] # 取出前三個特徵值，想降到幾維就取幾個

# 對特徵再歸一化
for i in range(3):
    L = LA.norm(p1[:,i])
    p1[:,i] = p1[:,i]/L

U = np.matmul(np.transpose(xm), p1) # 將測試影像和特徵值做矩陣相乘，得到降維後的數據

# ====================================================================================================
#    projection
# ====================================================================================================

print("="*100)
error = 0

tr_coe = np.matmul(np.transpose(U), np.transpose(xm)) # 訓練集投影
# print(tr_coe.shape)

print("\n｜來源｜判別｜")

# 逐一取出測試集的圖片
for i in range(len(ty)):   
    vec = tx[i]
    vec = vec - m1 # 零均值化
    test_coe = np.matmul(np.transpose(U), vec) # 測試集投影
    # print(test_coe.shape)

    # 如果有距離更短的就留下來
    minval = 999999999999999999999
    for j in range(len(x)): 
        dist = LA.norm(test_coe - tr_coe[:,j]) # 計算距離
        
        if (minval > dist):
            minval = dist
            idx = y[j] # 將目前距離最近的留下

    # 如果判別跟來源不同，錯誤記一分
    if idx != ty[i]:
        error = error + 1
    print("｜%4s｜%4s｜" %(ty[i],idx))

currect = (1 - (error/len(ty))) * 100
print("\n　正確率：%6.2f %%" %currect)

endTime = time.time()
elapsedTime = endTime - startTime
print("花費時間：%6.2f 秒\n" %elapsedTime)