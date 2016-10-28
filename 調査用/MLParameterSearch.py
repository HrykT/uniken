# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 13:17:54 2016

@author: Tsukamoto
"""

import os
import MLProc as mlp

#共通変数・定数
curdir = os.getcwd()

####ステップ数別学習用ファイル加工####
def learn_data_proc():
    for step in (5,10,20,30,40,50,60,70,80,90,100):
        mlp.ini_parts(
                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_1_関_発注_胸_20161020165951.csv")
                 ,[os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_1_関_発注_腕_20161020165950.csv")
                   ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_1_関_発注_腰_20161020165948.csv")]
                ,os.path.join(curdir,u"datas",u"concat_parts", u"step_n"  ,u"1-1_関_発注_uniken_step%d_processed.csv" % step)
                 ,"order",step)
    
        mlp.ini_parts(
                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_2_関_棚卸_胸_20161020170752.csv")
                 ,[os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_2_関_棚卸_腕_20161020170749.csv")
                   ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_2_関_棚卸_腰_20161020170747.csv")]
                ,os.path.join(curdir,u"datas",u"concat_parts", u"step_n"  ,u"1-2_関_棚卸_uniken_step%d_processed.csv" % step)
                 ,"tanaoroshi",step)
    
        mlp.ini_parts(
                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_3_関_品出し_胸_20161020171357.csv")
                 ,[os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_3_関_品出し_腕_20161020171356.csv")
                   ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_3_関_品出し_腰_20161020171354.csv")]
                ,os.path.join(curdir,u"datas",u"concat_parts", u"step_n"  ,u"1-3_関_品出し_uniken_step%d_processed.csv" % step)
                 ,"display",step)
    
        mlp.ini_parts(
                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_4_関_事務_胸_20161020172042.csv")
                 ,[os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_4_関_事務_腕_20161020172041.csv")
                   ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_4_関_事務_腰_20161020172040.csv")]
                ,os.path.join(curdir,u"datas",u"concat_parts", u"step_n"  ,u"1-4_関_事務_uniken_step%d_processed.csv" % step)
                 ,"officework",step)
    
        mlp.ini_parts(
                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_5_関_掃除_胸_20161020172749.csv")
                 ,[os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_5_関_掃除_腕_20161020172747.csv")
                   ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_5_関_掃除_腰_20161020172745.csv")]
                ,os.path.join(curdir,u"datas",u"concat_parts", u"step_n"  ,u"1-5_関_掃除_uniken_step%d_processed.csv" % step)
                 ,"cleaning",step)
    
        mlp.ini_parts(
                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_7_関_歩行_胸_20161021121819.csv")
                 ,[os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_7_関_歩行_腕_20161021121819.csv")
                   ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_7_関_歩行_腰_20161021121819.csv")]
                ,os.path.join(curdir,u"datas",u"concat_parts", u"step_n"  ,u"1-7_関_歩行_uniken_step%d_processed.csv" % step)
                 ,"walk",step)
        print(u"ステップ%d　処理終了" % step)
#実行
#learn_data_proc()

####ステップ数別予測用ファイル加工####
def test_data_proc():
    import pandas as pd
    mixlabel = pd.DataFrame([
    ["2016/10/20 17:36:58",
     "2016/10/20 17:38:50",
    "2016/10/20 17:39:18",
    "2016/10/20 17:40:36",
    "2016/10/20 17:41:04",
    "2016/10/20 17:42:25",
    "2016/10/20 17:42:56",
    "2016/10/20 17:43:38"],
    ["officework",
     "walk",
     "nimotsu",
     "walk",
     "display",
     "service",
     "walk",
     "order"]
    ]).T
    for step in [5]:
    #,10,20,30,40,50,60,70,80,90,100]:
        mlp.ini_parts(
                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_6_関_事務荷物品出し発注_胸_20161020173701.csv")
                 ,[os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_6_関_事務荷物品出し発注_腕_20161020173659.csv")
                   ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_6_関_事務荷物品出し発注_腰_20161020173657.csv")]
                ,os.path.join(curdir,u"datas",u"concat_parts", u"step_n_test"  ,u"1_6_関_事務荷物品出し発注_step%d_processed.csv" % step)
                 ,"mix",step, mix_label=mixlabel)
        print(u"ステップ%d　処理終了" % step)

test_data_proc()