# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:12:04 2016

@author: tsukamoto
"""

class MyPCA:
    #主成分分析のためのクラス
    def __init__(self, target):
        #初期化時にtarget（元データ）を正規化する
        from sklearn.preprocessing import StandardScaler
        std = StandardScaler()
        self.std_data = std.fit_transform(target)
        #pcaオブジェクト初期化
        self.pca = None
    def pca_fit(self,n=2):
        from sklearn.decomposition import PCA
        p = PCA(n_components=n)
        return p.fit_transform(self.std_data)
        
    def explained_variance_ratio(self, n=[1]):
        #圧縮先の次元数を複数指定して、累積寄与率を比較する
        from sklearn.decomposition import PCA
        from numpy import cumsum
        print(u"n,累積寄与率")
        for i in n:
            self.pca = PCA(n_components=i)
            self.pca.fit(self.std_data)
            kiyo = self.pca.explained_variance_ratio_
            #print("%d,%.6f" % (i,cumsum(kiyo)))
            print("%d,%.6f" % (i,cumsum(kiyo)[::-1][0]))
        