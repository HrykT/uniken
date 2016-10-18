# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:01:39 2016

@author: tsukamoto
"""

import Single_Models as md
import Plot_predict as plop
import numpy as np
import pandas as pd
import os

####################Single_Models 使い方サンプル###########################

#使用するアルゴリズムの辞書リスト
algs =  {
           u"svm_linear" : md.svm_linear
         , u"svm_rbf" : md.svm_rbf
         , u"ｋ_neighbors" : md.kNeighbors
         ,u"svm_poly": md.svm_poly
         , u"logistic_regression" : md.logistic_regression
        }
        
#加工済み学習用データを分類器に渡せる形に加工する
def proc_for_fit(dates):
    #特定ラベルのみを貼り付けた各学習データファイルを結合する
    alldata = pd.concat(dates,axis=0)
    #ラベルデータを切り離して返す
    label = alldata["label"]
    #日時も使えないのでドロップ
    return label, alldata.drop(["label","datetime"], axis=1)

#加工済みデータから学習用データに加工実行
curdir = os.getcwd()
files = [
         pd.read_csv(os.path.join(curdir,u"datas",u"sample_proc_data"
                ,u"1-1_関_発注_腰_uniken_processed.csv"))
        ,pd.read_csv(os.path.join(curdir,u"datas",u"sample_proc_data"
                ,u"1-1_関_品出し_腰_uniken_processed.csv"))
            ]

y,X = proc_for_fit(files)

#PCAで特徴抽出
import MyPCA as mpca
myp = mpca.MyPCA(X)
X_pca = myp.pca_fit(n=5)

#指定のテストレートで各アルゴリズムを使い学習・精度確認
for algkey in algs.keys():
    my_clf = algs[algkey](X,y,0.3)
    my_clf.name = algkey
    my_clf.fit()
    my_clf.show_score()