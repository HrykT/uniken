# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:35:56 2016

@author: tsukamoto
"""

import InitDataProcess as inip
import pandas as pd
import numpy as np
import os


####################InitDataProcess 使い方サンプル###########################

#統計量計算対象の列名
stat_target =  ["X_acceleration"
               ,"Y_acceleration"
               ,"Z_acceleration"
               ,"X_gravity"
               ,"Y_gravity"
               ,"Z_gravity"
               ,"X_angular_velocity"
               ,"Y_angular_velocity"
               ,"Z_angular_velocity"
               ,"azimuth"
               ,"pitch"
               ,"roll"
               ]

#使用しない列名
del_column = ["X_magnetic_field"
                  ,"Y_magnetic_field"
                  ,"Z_magnetic_field"
                  ,"X_linear_acceleration"
                  ,"Y_linear_acceleration"
                  ,"Z_linear_acceleration"]

name_func = {
#     "_Max" : np.max #最大値
#    ,"_Min" : np.min #最小値
#    ,
     "_Avg" : np.average #平均値
    ,"_Var" : np.var #分散 
    }

#単独ファイルの加工
def ini_single(before,after,label,n):
    #データ読み込み
    dt = pd.read_csv(before, sep=",", encoding='shift-jis')
    #読み込みデータ、基本列名、ラベル名で初期化
    initdata = inip.InitDataProc(dt,stat_target
                                 ,label,n
                                 ,del_unitchar_cols=["pitch","roll","azimuth"])
    
    #統計量追加、ファイル出力
    initdata.add_statvals(name_func)
    #正解ラベル追加
    initdata.add_label()
    #ファイル保存
    initdata.output(after)
    
#ファイルの結合を含む加工
def ini_parts(basefile,parts,after,label,n):          
    #ベースデータ読み込み
    base_dt = pd.read_csv(basefile, sep=",", encoding='shift-jis')
    #結合データ読み込み
    parts_dt = (pd.read_csv(pt, sep=",", encoding='shift-jis') for pt in parts)
    #読み込みデータ、基本列名、ラベル名で初期化
    #ベース
    inidata_base = inip.InitDataProc(base_dt.drop(del_column, axis=1)
                                 ,stat_target
                                 ,label,n
                                 ,del_unitchar_cols=["pitch","roll","azimuth"])
    #datetimeを秒まで切り捨て、秒ごとのインデックス追加
    inidata_base.add_datetime_idx()
    #統計量追加
    inidata_base.add_statvals(name_func)
    #列名書き換え
    inidata_base.update_columnnames("_1")
    
    #パーツ
    initdata_parts = (inip.InitDataProc(p.drop(del_column, axis=1)
                                 ,stat_target
                                 ,label,n
                                 ,del_unitchar_cols=["pitch","roll","azimuth"])
                        for p in parts_dt)

    idx=2
    merge_datas = []
    for p in initdata_parts:
        p.add_datetime_idx()
        p.add_statvals(name_func)
        p.update_columnnames("_" + str(idx))
        merge_datas.append(p.processed_data)
        idx += 1
        
    #結合
    inidata_base.merge_partsdata(merge_datas)
    #正解ラベル追加
    inidata_base.add_label()
    #余分データ削除
    inidata_base.drop_record_first_last(second=10)
    #ファイル保存
    inidata_base.output(after)


####################実行###########################
curdir = os.getcwd()

#単独
ini_single(os.path.join(curdir,u"datas",u"20161006_sample_koshi"
                ,u"1-1_関_発注_腰_uniken_20161006165941.csv")
    ,os.path.join(curdir,u"datas","sample_proc_data",u"1-1_関_発注_腰_uniken_processed.csv")
    ,"order", 25)

#結合
ini_parts(os.path.join(curdir,u"datas",u"20161006_sample_koshi"
                ,u"1-1_関_発注_腰_uniken_20161006165941.csv")
         ,[os.path.join(curdir,u"datas",u"20161006_sample_mune"
                ,u"1-1_関_発注_胸_uniken_20161006165944.csv")
           ,os.path.join(curdir,u"datas",u"20161006_sample_ude"
                ,u"1-1_関_発注_腕_uniken_20161006165941.csv")]
         ,os.path.join(curdir,u"datas",u"sample_proc_data"
                  ,u"1-1_関_発注_uniken_processed.csv")
         ,"order",25)