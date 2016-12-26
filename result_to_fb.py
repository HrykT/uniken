#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


'''
  与えられたリストの中に、最も多く存在する要素を返す
  (最大の数の要素が複数ある場合、pythonのsetで先頭により近い要素を返す)
'''
def maxElem( lis ):

    L = lis[:]#copy
    S = set(lis)
    S = list(S)
    MaxCount=0
    ret='nothing...'

    for elem in S:
        c=0
        while elem in L:
            ind = L.index(elem)
            foo = L.pop(ind)
            c+=1
        if c>MaxCount:
            MaxCount=c
            ret = elem
    return ret

# main
# 読み込んだファイルを配列にし、指定した行のまとまりの中で
# 最頻の項目で他をつぶす

# 判定結果を読み込む
fname = 'origin_result_rbf_20161202for20161128.csv'
Y = np.loadtxt(fname,delimiter=",",dtype="str")[:,1]

N = Y.size
print N
# 何行を1つにするか
M = 4500 

# 0からNまでM刻みでループ(0,M,2M,3M,...,N)
res = []
for i in range(0,N,M):
    # 指定した数だけ出力
    # 指定した行の中で最頻の項目を出力
    res.append(maxElem(Y[i:i+M].tolist()))

np.savetxt("t_" + fname, res, fmt="%s")