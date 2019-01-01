#!/usr/bin/env python
# coding: utf-8

# 先看這張圖 https://img-blog.csdn.net/20171022165350939


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

PATH = 'C:/Users/wei/Documents/ML/Faces' # 指定圖像資料夾路徑
facesDir = os.listdir(PATH)  # 開啟圖像資料夾
trainDataFiles = open('train.txt', 'w') # 以寫入模式開啟 train.txt
testDataFiles = open('test.txt', 'w') # 以寫入模式 test.txt

print("已讀取指定資料夾 %s，包含 %s 個檔案" %(PATH, len(facesDir)))

# Len = 取得內容的個數
# 迴圈是要取得 Faces 資料夾內的所有資料夾
for i in range(len(facesDir)):
    d2 = os.listdir (PATH + '/%s' %(facesDir[i])) # 進入 Faces 資料夾再下一層的各資料夾

    for j in range(len(d2)-3): # 注意這邊減去一張（最後一張不拿），是為了保留等等測試集要用的圖片
        str1 = PATH + '/%s/%s' %(facesDir[i], d2[j]) # 取得每個子資料夾的內容檔案路徑
        print("正在寫入第 %s 人的圖像檔案路徑至訓練集 Train.txt：%s" %(i + 1 , str1))
        trainDataFiles.write("%s %d\n" % (str1, i)) # 將每個內容檔案路徑都存放到 train.txt，並標註是第幾個人

    for h in range(3):
        str1 = PATH + '/%s/%s' %(facesDir[i], d2[-h]) # 取得每個資料夾中的最後一張圖片的路徑
        print("正在寫入第 %s 人的圖像檔案路徑至測試集  Test.txt：%s" %(i + 1, str1))
        testDataFiles.write("%s %d\n" % (str1, i)) # 將每個資料夾最後一張圖片的路徑都存放到 test.txt，並標註是第幾個人

# 關閉 train.txt 和 test.txt
trainDataFiles.close() 
testDataFiles.close()

# 上面這個區塊的目的在於：將圖片庫中的每個資料夾逐一打開，並將最後一張放到測試集的文字檔、以及將其他張照片放到訓練集的文字檔

print("已成功將資料寫入至 train.txt 和 test.txt\n","="*100)

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
        im1 = cv2.resize(im1, (32,32)) # 調整圖片尺寸
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
#    LDA
# ====================================================================================================

m = np.mean(x, 0) # 全局平均
xm = x - m

# 初始化 SW 和 SB
sw = np.zeros((30,30), dtype=np.float32)
sb = np.zeros((30,30), dtype=np.float32)

# 取出對應標籤的照片
labels = max(y)+1
for i in range(labels):
    sub_x = x[y==i,:] # (19,4096)
    print("正在處理 第 %s 人 的圖像矩陣" %(i+1))
    
    mi = np.mean(sub_x, 0) # 該人物的平均

    for j in range(len(sub_x)):

        xmi = sub_x[j] - mi # 零均值化
        swi = np.cov(xmi) # 算個人的 SW 並加進去
        sw += swi
    
    mt = mi - m # 該人物平均扣除全局平均
    sbi = 2 * np.matmul(mt, np.transpose(mt)) # 算個人的 SB 並加進去
    sb += sbi

# 奇異矩陣分解，不然會跳錯誤
print("奇異矩陣無法直接處理，正在進行分解")
U, S, V = LA.svd(sw)
S = np.diag(S)
SWnew = V.dot(LA.pinv(S)).dot(U.T)
A = SWnew.dot(sb)

print("分解完成，正在計算特徵矩陣向量")
w, v = LA.eig(A) # 計算 特徵矩陣向量

print("\n特徵矩陣向量\n特徵值：", w.shape)
print("特徵向量：", v.shape)

v = -np.sort(-v, axis = 1) # 按列遞減排序，若未排序正確率會下降很多

p1 = v[:, 0:3] # 取出前三個特徵值，想降到幾維就取幾個

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
    vec = vec - m # 零均值化
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