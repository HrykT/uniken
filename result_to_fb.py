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
Y = np.loadtxt('result_test.txt')
N = Y.size

# 何行を1つにするか
M = 25 

# 0からNまでM刻みでループ(0,M,2M,3M,...,N)
for i in range(0,N,M):
    # 指定した数だけ出力
    for j in range(M):
        # 元の判定結果の数を超えないように
        if i + j < N:
            # 指定した行の中で最頻の項目を出力
            print maxElem(Y[i:i+M].tolist())
