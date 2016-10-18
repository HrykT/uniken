# -*- coding: utf-8 -*-
"""
Created on Fri Sep 09 00:22:10 2016

@author: tsukamoto
"""

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

#決定領域の色分けグラフ
#使い方メモ
#x_cmb = np.vstack((svmrbf.X_train,svmrbf.X_test))[0:100]
#y_cmb = svmrbf.class_la.inverse_transform(
#             np.hstack((svmrbf.y_train,svmrbf.y_test))[0:100])
#import matplotlib.pyplot as plt
#import plot_predict as plop
#plop.plot_decision_regions(X=x_cmb, y=y_cmb,classifier=clf)
#plt.xlabel(u"x_max")
#plt.xlabel(u"z_max")
#plt.legend(loc = "upper left")
#plt.show()
def plot_decision_regions(X,y,classifier,resolution=0.05):
    """X:2つまでの特徴量。pythonリストかnumpy行列型。pandasデータフレームは使えない
       y:ラベル。トレーニング用とテスト用を分けたものを結合しておく"""
    #マーカー、色のリスト
    markers = ("o","x","^","s","v")
    colors = ("red","blue","lightgreen","gray","cyan")
    #インスタンス
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    #領域の最大、最小をプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    #グリッドポイントの作成
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max ,resolution),
                           np.arange(x2_min, x2_max ,resolution))
    
    #グリッドポイントを一次元化して予測
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    #予測結果を元のサイズに戻す
    Z = Z.reshape(xx1.shape)

    #等高線のプロット
    plt.contourf(xx1, xx2, Z,alpha=0.4, cmap=cmap)
    
    #軸の範囲設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    #クラスごとにサンプルをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], 
                    alpha = 0.8,c=cmap(idx),
                    marker=markers[idx], label=cl)