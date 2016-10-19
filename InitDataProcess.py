# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 19:23:34 2016

@author: tsukamoto
"""

import pandas as pd
import numpy as np

class InitDataProc:
    def __init__(self, base, axisNames, label, n=5, del_unitchar_cols=[""]):
        #加工前データ
        self.base_data = base
        #単位文字削除
        self.delete_unitstr(del_unitchar_cols)
        #統計計算のベースにする、元データ行数-n+1の連番ベクトル
        self.base_data_rows = pd.Series(np.arange(0,len(self.base_data)-n+1))
        #元データの各軸の列名
        self.axisNames = axisNames
        #加工後データ(末尾n+1行は使わない)
        self.processed_data = self.base_data[0:len(self.base_data)-n+1]
        #結果ラベル
        self.label = label
        #統計量算出に使用するレコード数
        self.n = n

    #各向き加速度要素のnレコードから任意の関数で値を計算・列追加する
    def add_statval(self,name,func):
        for ax in self.axisNames:
            val = ["%.6f" % func(self.base_data.loc[num:num+self.n-1 ,ax])
                            for num in self.base_data_rows]
            self.processed_data.loc[:,ax + name]= val
            self.processed_data.loc[:,ax + name].astype(np.float64)
    #複数の統計量をまとめて追加する
    def add_statvals(self,names_funcs):
        for n,f in sorted(names_funcs.items()):
            self.add_statval(n,f)
    #ラベル列を追加する（教師用データに使用 ひとつのDataFrameすべてに同じラベルを貼る）
    def add_label(self):
        lbl = pd.DataFrame([[self.label] * len(self.processed_data)]).T
        lbl.columns = ["label"]
        self.processed_data = pd.concat([lbl,self.processed_data],axis=1)
    #単位文字を削除する
    def delete_unitstr(self,cols):
        for col in cols:
            if col in self.base_data.columns \
             and isinstance((self.base_data.loc[:,col][1]),unicode):
                 #指定の列名が含まれ、頭のデータが文字の場合のみ処理
                self.base_data.loc[:,col].str.encode('shift-jis')
                self.base_data.loc[:,col] = \
                    self.base_data.loc[:,col].str.replace(u'°',u'') 
                #型を戻す
                self.base_data.loc[:,col] = self.base_data.loc[:,col].astype(np.float64)
    #列名を一括書き換え
    def update_columnnames(self,addstr):
        self.processed_data.columns = \
            (c + addstr for c in self.processed_data.columns)
        #datetime,datetime_indexのみ戻す
        self.processed_data.rename(
            columns = {'datetime' + addstr : 'datetime'
                      ,'datetime_index' + addstr : 'datetime_index'}
                                    ,inplace=True)
    #baseデータをメインとして、部位ごとのセンサーデータを結合する
    def merge_partsdata(self,parts_datas,fill_type='f'):
        """fill_type : f -> 前方埋め
                       z -> ゼロ埋め
                       m -> 各列平均値埋め
        """
        for d in parts_datas:
            self.processed_data = self.processed_data.merge(d,
                                            on=['datetime','datetime_index'],
                                            how="outer")
        if fill_type == 'm':
            #欠損値の平均値埋め
            target_clms = (trg for trg in self.processed_data.columns
                            if trg not in ['datetime','datetime_index'])
            for c in target_clms:
                m = self.processed_data.loc[:,c].astype(np.float).mean()
                self.processed_data.loc[:,c] = \
                        self.processed_data.loc[:,c].fillna(m)
        elif fill_type == 'f':
            #欠損値の前方埋め
            self.processed_data = self.processed_data.fillna(method='ffill')
        elif fill_type == 'z':
            #欠損値の前方埋め
            self.processed_data = self.processed_data.fillna(0)
    #秒までのグループ内でのインデックスを追加する
    def add_datetime_idx(self):
       #日時を秒までに削除
        self.processed_data.loc[:,'datetime'] = \
            self.processed_data.loc[:,'datetime'].str[:19]
        #秒までが同じレコードに連番をつける
        grp_idx = 0
        grp_idx_clm = []
        before_time = ""
        for i, row in self.processed_data.iterrows():
            if not row['datetime'] == before_time:
                grp_idx = 1
            else:
                grp_idx += 1
            before_time = row['datetime']
            grp_idx_clm.append(grp_idx)
        self.processed_data.loc[:,'datetime_index'] = grp_idx_clm
    #最初と最後のn秒分のデータを捨てる 200msごとのレコードとする
    def drop_record_first_last(self,second=5):
        rec_per_sec = 5
        self.processed_data = \
            self.processed_data[second + rec_per_sec :
                len(self.processed_data.index) - second + rec_per_sec]
    #ｃｓｖに書き出す
    def output(self,filepath):
        self.processed_data.to_csv(filepath, index=False, encoding='shift-jis')