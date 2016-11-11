# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 17:36:04 2016

@author: tsukamoto
"""

import InitDataProcess as inip
import Single_Models as md
import Plot_predict as plop
import pandas as pd
import numpy as np
import os

#ファイル初期処理　毎回やらなくてよし

#統計量計算対象の列名
stat_target =  [   "X_acceleration"
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

del_column = ["X_magnetic_field"
                  ,"Y_magnetic_field"
                  ,"Z_magnetic_field"
]

name_func = {
     "_Max" : np.max #最大値
    ,"_Min" : np.min #最小値
    ,"_Avg" : np.average #平均値
    ,"_Var" : np.var #分散 
    }
    
def ini(before,after,label,n):
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

#ファイル初期処理　複数ファイルの結合版　毎回やらなくてよし
def ini_parts(basefile, parts1, parts2,after,label,n ,mix_label=None):
    #ベースデータ読み込み
    base_dt = pd.read_csv(basefile, sep=",", encoding='shift-jis')
    #結合データ読み込み
    parts_dt1 = pd.read_csv(parts1, sep=",", encoding='shift-jis')
    parts_dt2 = pd.read_csv(parts2, sep=",", encoding='shift-jis')
    ini_merge_proc(base_dt, parts_dt1, parts_dt2, after, label, n, mix_label)
    
    
#同じファイルの前半を学習用、後半をテスト用に別々に加工
def ini_parts_split(basefile, parts1, parts2, after_learn, after_test, label, n):
    #ベースデータ読み込み
    base_dt = pd.read_csv(basefile, sep=",", encoding='shift-jis')
    base_learn = base_dt[:len(base_dt.index) // 2]
    base_test = base_dt[(len(base_dt.index) // 2) + 1 :].reset_index(drop=True)
    #結合データ読み込み
    parts_dt1 = pd.read_csv(parts1, sep=",", encoding='shift-jis')
    parts1_learn =parts_dt1.loc[:len(parts_dt1.index) // 2]
    parts1_test = parts_dt1.loc[(len(parts_dt1.index) // 2) + 1:].reset_index(drop=True)

    parts_dt2 = pd.read_csv(parts2, sep=",", encoding='shift-jis')
    parts2_learn =parts_dt2.loc[:len(parts_dt2.index) // 2]
    parts2_test = parts_dt2.loc[(len(parts_dt2.index) // 2) + 1:].reset_index(drop=True)
    #学習データ処理
    ini_merge_proc(base_learn,parts1_learn, parts2_learn, after_learn,label,n)

    #テストデータ処理
    ini_merge_proc(base_test, parts1_test, parts2_test, after_test, label,n)    

def ini_merge_proc(base_dt,parts1,parts2,after,label,n ,mix_label=None):
    #複数ファイル結合初期処理用の共通関数
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
    initdata_parts1 = inip.InitDataProc(parts1.drop(del_column, axis=1)
                                 ,stat_target
                                 ,label,n
                                 ,del_unitchar_cols=["pitch","roll","azimuth"])

    initdata_parts2 = inip.InitDataProc(parts2.drop(del_column, axis=1)
                                 ,stat_target
                                 ,label,n
                                 ,del_unitchar_cols=["pitch","roll","azimuth"])
    merge_datas = []
    initdata_parts1.add_datetime_idx()
    initdata_parts1.add_statvals(name_func)
    initdata_parts1.update_columnnames("_2")
    merge_datas.append(initdata_parts1.processed_data)

    initdata_parts2.add_datetime_idx()
    initdata_parts2.add_statvals(name_func)
    initdata_parts2.update_columnnames("_3")
    merge_datas.append(initdata_parts2.processed_data)
        
    #結合
    inidata_base.merge_partsdata(merge_datas,fill_type='m')
    #正解ラベル追加
    inidata_base.add_label()
    #ソート
    inidata_base.sort_data()
    #混合ラベルがあれば混合ラベル
    if mix_label is not None:
        inidata_base.add_label_mix(mix_label)
    #余分データ削除
    inidata_base.drop_record_first_last(second=5)
    #ファイル保存
    inidata_base.output(after)

#初期処理 未来レコードの横連結版
def ini_concat_nextrec(basefile, parts1, parts2,after,label,n ,mix_label=None):
    #ベースデータ読み込み
    base_dt = pd.read_csv(basefile, sep=",", encoding='shift-jis')
    #結合データ読み込み
    parts1 = pd.read_csv(parts1, sep=",", encoding='shift-jis')
    parts2 = pd.read_csv(parts2, sep=",", encoding='shift-jis')
    
    #読み込みデータ、基本列名、ラベル名で初期化
    #ベース
    inidata_base = inip.InitDataProc(base_dt.drop(del_column, axis=1)
                                 ,stat_target
                                 ,label,n
                                 ,del_unitchar_cols=["pitch","roll","azimuth"])
    #datetimeを秒まで切り捨て、秒ごとのインデックス追加
    inidata_base.add_datetime_idx()
    #列名書き換え
    inidata_base.update_columnnames("_1")
    
    #パーツ
    initdata_parts1 = inip.InitDataProc(parts1.drop(del_column, axis=1)
                                 ,stat_target
                                 ,label,n
                                 ,del_unitchar_cols=["pitch","roll","azimuth"])

    initdata_parts2 = inip.InitDataProc(parts2.drop(del_column, axis=1)
                                 ,stat_target
                                 ,label,n
                                 ,del_unitchar_cols=["pitch","roll","azimuth"])
    merge_datas = []
    initdata_parts1.add_datetime_idx()
    initdata_parts1.update_columnnames("_2")
    merge_datas.append(initdata_parts1.processed_data)

    initdata_parts2.add_datetime_idx()
    initdata_parts2.update_columnnames("_3")
    merge_datas.append(initdata_parts2.processed_data)
        
    #結合
    inidata_base.merge_partsdata(merge_datas,fill_type='m')
    #ソート
    inidata_base.sort_data()
    #未来レコード横結合
    inidata_base.add_column_next_row(n=5)
    #正解ラベル追加
    inidata_base.add_label()
    #混合ラベルがあれば混合ラベル
    if mix_label is not None:
        inidata_base.add_label_mix(mix_label)
    #余分データ削除
    inidata_base.drop_record_first_last(second=5)
    #ファイル保存
    inidata_base.output(after)
    
#学習用データを分類器に渡せる形に加工する
def proc_for_fit(dates):
    #特定ラベルのみを貼り付けた各学習データファイルを結合する
    alldata = pd.concat(dates,axis=0)
    #ラベルデータを切り離して返す
    label = alldata["label"]
    #日時も使えないのでドロップ
    return label, alldata.drop(["label","datetime","datetime_index"], axis=1)

#学習＋分類器保存
def run_ml(model_class, feuture_values, labels, test_rate, clf_name,
               path, use_saved_clf=False, show_plot=False):
#    if show_plot:
#        #特徴量をグラフ用２次元データに加工 …おもたいのでいったんやめ
#        from sklearn.decomposition import PCA
#        pca = PCA(n_components=2)
#        feuture_values = pca.fit_transform(feuture_values)
    model_inst = model_class(feuture_values,labels,test_rate)
    model_inst.name = clf_name
    if use_saved_clf:
        #学習済み分類器読み込み
        load = md.modelIO()
        classifier = load.load_model(path)
    else:
        #新規に学習
        classifier = model_inst.fit()
        #保存
        save = md.modelIO()
        save.save_model(classifier,path)
    model_inst.show_score()
#    if clf_name == "logistic_regression":
#        #ロジスティック回帰の場合のみ変数の影響度を表示
#        model_inst.show_coefficients()
    if show_plot:
        #グラフにプロット
        showplot(model_inst, classifier, 200, "", "")

#複数の分類器を返す＋保存
def run_mls(model_instance,save_path_root):
    fld_clf = model_instance.fit()
    basename = model_instance.name
    for fld,clf in fld_clf.items():
        model_instance.name = basename + fld
        model_instance.show_score(clf)
        #保存
        save = md.modelIO()
        save.save_model(clf,save_path_root + fld + "/" + fld)
    return fld_clf

#交差検証
def run_crossval(model_class, feuture_values, labels,
                 test_rate, clf_name):
    clf = model_class(feuture_values,labels,test_rate)
    clf.name = clf_name
    clf.closs_vld(k=10)

#予測をグラフにプロット
def showplot(single_model, clf, count, xlbl, ylbl):
    """
    single_model -> single_model_baseクラスのインスタンス
    clf　-> 使用する分類器
    count　->　0~何行までのデータをプロットするか
    xlbl　-> x軸ラベル名
    ylbl -> y軸ラベル名
    """
    #特徴量を行方向結合
    x_cmb = np.vstack((single_model.X_test
                    ,single_model.X_train))[:count]
    #ラベルを列方向結合(正解ラベル名称が横一列に並ぶ)
    y_cmb = single_model.class_la.inverse_transform(
                 np.hstack((single_model.y_test
                          , single_model.y_train
                         ))[:count])
    import matplotlib.pyplot as plt
    plop.plot_decision_regions(X=x_cmb, y=y_cmb,classifier=clf)
    plt.title(single_model.name)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.legend(loc = "upper left"
       		, bbox_to_anchor=(0.5,-0.1) 
 			, ncol=2
            )
    plt.show()

##########ここから実際に動かす用の記述############
curdir = os.getcwd()
step = 25

#使用するアルゴリズムの辞書リスト
algs =  {
           u"svm_linear" : md.svm_linear
         , u"svm_rbf" : md.svm_rbf
         , u"ｋ_neighbors" : md.kNeighbors
         , u"svm_poly": md.svm_poly
         , u"logistic_regression" : md.logistic_regression
         , u"decision_tree" : md.decision_tree
         , u"k_means" : md.k_means
         , u"RandomForest" : md.RandomForest
         , u"GBDT" : md.GBDT
         #, u"Adaboost" : md.Adaboost
         , u"neural_net" : md.neural_net
        }

#最適化に用いるアルゴリズム別パラメータリスト
alg_params = {
           u"svm_linear" : {'C': [1, 10, 100, 1000], 'kernel': ['linear']}
           
         , u"svm_rbf" : {'C': [10, 100], 'kernel': ['rbf']
                         , 'gamma': [0.001, 0.1, 1.0]}
                         
         , u"ｋ_neighbors" : {u'n_neighbors': [1,3,5]
                             , u'weights': ['uniform','distance']}
                             
         ,u"svm_poly": {'C': [100, 1000], 'kernel': ['poly']
                 , 'degree': [3, 4], 'gamma': [0.001, 0.1, 1.0]}
                 
         , u"logistic_regression" : {'C': [1, 10, 100, 1000]}
         
         , u"decision_tree" : {'max_depth' : [2,3,4,5,None]}
         
         , u"k_means" : {'n_clusters' : [2,4,8,10]}
         
         , u"RandomForest" : {'n_estimators' : [3,5,10]}
         
         , u"GBDT" : {'n_estimators' : [10,50,100], 'max_depth' : [1,3,5],
                      'lerning_rate' : [0.1, 0.2, 0.3]}
         
         #, u"Adaboost" : {}
         
         , u"neural_net" :{'activation' : ['identity', 'logistic', 'tanh', 'relu']
                           ,'solver' : ['lbfgs', 'sgd', 'adam']}
         }
#使用する特徴量
feuture_value_columns = ["label","datetime","X_acceleration","Y_acceleration",
                         "Z_acceleration","X_acceleration_Avg",
                         "Y_acceleration_Avg","Z_acceleration_Avg",
                         "X_acceleration_Max","Y_acceleration_Max",
                         "Z_acceleration_Max","X_acceleration_Min",
                         "Y_acceleration_Min","Z_acceleration_Min",
                         "X_acceleration_Var","Y_acceleration_Var",
                         "Z_acceleration_Var"]
        
###生データ初期処理実行###
#ini(os.path.join(curdir,u"datas",u"seki1",u"clotheshorse.csv")
#    ,os.path.join(os.path.join(curdir,u"datas",u"seki1",u"clotheshorse_2.csv")
#    ),"monohoshi", 25)
#ini_parts(os.path.join(curdir,u"datas",u"20161006_sample_koshi"
#                ,u"1-1_関_発注_腰_uniken_20161006165941.csv")
#         ,[os.path.join(curdir,u"datas",u"20161006_sample_mune"
#                ,u"1-1_関_発注_胸_uniken_20161006165944.csv")
#           ,os.path.join(curdir,u"datas",u"20161006_sample_ude"
#                ,u"1-1_関_発注_腕_uniken_20161006165941.csv")]
#         ,os.path.join(curdir,u"datas",u"concat_parts"
#                  ,u"1-1_関_発注_uniken_step%d_processed.csv" % step)
#         ,"order",step)
#ini_parts(os.path.join(curdir,u"datas",u"20161006_sample_koshi"
#                ,u"1-2_関_品出し_腰_uniken_20161006170642.csv")
#         ,[os.path.join(curdir,u"datas",u"20161006_sample_mune"
#                ,u"1-2_関_品出し_胸_uniken_20161006170644.csv")
#           ,os.path.join(curdir,u"datas",u"20161006_sample_ude"
#                ,u"1-2_関_品出し_腕_uniken_20161006170642.csv")]
#         ,os.path.join(curdir,u"datas",u"concat_parts"
#                  ,u"1-2_関_品出し_uniken_step%d_processed.csv" % step)
#         ,"display",step)
#ini_parts(os.path.join(curdir,u"datas",u"20161006_sample_koshi"
#                ,u"1-3_関_棚卸_腰_uniken_20161006171459.csv")
#         ,[os.path.join(curdir,u"datas",u"20161006_sample_mune"
#                ,u"1-3_関_棚卸_胸_uniken_20161006171502.csv")
#           ,os.path.join(curdir,u"datas",u"20161006_sample_ude"
#                ,u"1-3_関_棚卸_腕_uniken_20161006171459.csv")]
#         ,os.path.join(curdir,u"datas",u"concat_parts"
#                  ,u"1-3_関_棚卸_uniken_step%d_processed.csv" % step)
#         ,"tanaoroshi",step)
#ini_parts(os.path.join(curdir,u"datas",u"20161006_sample_koshi"
#                ,u"1-4_関_荷物整理_腰_uniken_20161006172157.csv")
#         ,[os.path.join(curdir,u"datas",u"20161006_sample_mune"
#                ,u"1-4_関_荷物整理_胸_uniken_20161006172159.csv")
#           ,os.path.join(curdir,u"datas",u"20161006_sample_ude"
#                ,u"1-4_関_荷物整理_腕_uniken_20161006172156.csv")]
#         ,os.path.join(curdir,u"datas",u"concat_parts"
#                  ,u"1-4_関_荷物整理_uniken_step%d_processed.csv" % step)
#         ,"nimotsu",step)
#ini_parts(os.path.join(curdir,u"datas",u"20161006_sample_koshi"
#                ,u"1-5_関_荷物運搬_腰_uniken_20161006173027.csv")
#         ,[os.path.join(curdir,u"datas",u"20161006_sample_mune"
#                ,u"1-5_関_荷物運搬_胸_uniken_20161006173030.csv")
#           ,os.path.join(curdir,u"datas",u"20161006_sample_ude"
#                ,u"1-5_関_荷物運搬_腕_uniken_20161006173027.csv")]
#         ,os.path.join(curdir,u"datas",u"concat_parts"
#                  ,u"1-5_関_荷物運搬_uniken_step%d_processed.csv" % step)
#         ,"carry",step)
#ini_parts(os.path.join(curdir,u"datas",u"20161006_sample_koshi"
#                ,u"1-6_関_歩行_腰_uniken_20161006173742.csv")
#         ,[os.path.join(curdir,u"datas",u"20161006_sample_mune"
#                ,u"1-6_関_歩行_胸_uniken_20161006173744.csv")
#           ,os.path.join(curdir,u"datas",u"20161006_sample_ude"
#                ,u"1-6_関_歩行_腕_uniken_20161006173742.csv")]
#         ,os.path.join(curdir,u"datas",u"concat_parts"
#                  ,u"1-6_関_歩行_uniken_step%d_processed.csv" % step)
#         ,"walk",step)
#ini_parts(os.path.join(curdir,u"datas",u"20161006_sample_koshi"
#                ,u"2-1_渡辺_発注_腰_uniken_20161006175918.csv")
#         ,[os.path.join(curdir,u"datas",u"20161006_sample_mune"
#                ,u"2-1_渡辺_発注_胸_20161006175921.csv")
#           ,os.path.join(curdir,u"datas",u"20161006_sample_ude"
#                ,u"2-1_渡辺_発注_腕_uniken_20161006175918.csv")]
#         ,os.path.join(curdir,u"datas",u"concat_parts"
#                  ,u"2-1_渡辺_発注_uniken_step%d_processed.csv" % step)
#         ,"order",step)
#ini_parts(os.path.join(curdir,u"datas",u"20161006_sample_koshi"
#                ,u"2-6_渡辺_歩行_腰_uniken_20161006174605.csv")
#         ,[os.path.join(curdir,u"datas",u"20161006_sample_mune"
#                ,u"2-6_渡辺_歩行_胸_uniken_20161006174608.csv")
#           ,os.path.join(curdir,u"datas",u"20161006_sample_ude"
#                ,u"2-6_渡辺_歩行_腕_uniken_20161006174606.csv")]
#         ,os.path.join(curdir,u"datas",u"concat_parts"
#                  ,u"2-6_渡辺_歩行_uniken_step%d_processed.csv" % step)
#         ,"walk",step)
        
#ini_parts(os.path.join(curdir,u"datas"  ,u"20161020_採取データ"  ,u"uniken_1_6_関_事務荷物品出し発注_胸_20161020173701.csv")
#         ,[os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_6_関_事務荷物品出し発注_腰_20161020173657.csv")
#           ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_6_関_事務荷物品出し発注_腕_20161020173659.csv")]
#         ,os.path.join(curdir,u"datas",u"concat_parts" , u"test", u"xxx.csv")
#         ,"mix",step)
#
#
##同データを分割
#ini_parts_split(os.path.join(curdir,u"datas"
#,u"20161020_採取データ"                              ,u"uniken_1_1_関_発注_胸_20161020165951.csv")
#,[os.path.join(curdir,u"datas",u"20161020_採取データ",u"uniken_1_1_関_発注_腕_20161020165950.csv")
#,os.path.join(curdir,u"datas",u"20161020_採取データ" ,u"uniken_1_1_関_発注_腰_20161020165948.csv")]
#,os.path.join(curdir,u"datas",u"concat_parts","learn",u"1-1_関_発注_uniken_step%d_processed_l.csv" % step)
#,os.path.join(curdir,u"datas",u"concat_parts","test" ,u"1-1_関_発注_uniken_step%d_processed_t.csv" % step)
#,"order",step)
#ini_parts_split(os.path.join(curdir,u"datas"
#,u"20161020_採取データ"                                 ,u"uniken_1_2_関_棚卸_胸_20161020170752.csv")
#,[os.path.join(curdir,u"datas",u"20161020_採取データ",u"uniken_1_2_関_棚卸_腕_20161020170749.csv")
#,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_2_関_棚卸_腰_20161020170747.csv")]
#,os.path.join(curdir,u"datas",u"concat_parts","learn" ,u"1-2_関_棚卸_uniken_step%d_processed_l.csv" % step)
#,os.path.join(curdir,u"datas",u"concat_parts","test"  ,u"1-2_関_棚卸_uniken_step%d_processed_t.csv" % step)
#,"tanaoroshi",step)
#ini_parts_split(os.path.join(curdir,u"datas"
#,u"20161020_採取データ"                                 ,u"uniken_1_3_関_品出し_胸_20161020171357.csv")
#,[os.path.join(curdir,u"datas",u"20161020_採取データ",u"uniken_1_3_関_品出し_腕_20161020171356.csv")
#,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_3_関_品出し_腰_20161020171354.csv")]
#,os.path.join(curdir,u"datas",u"concat_parts","learn" ,u"1-3_関_品出し_uniken_step%d_processed_l.csv" % step)
#,os.path.join(curdir,u"datas",u"concat_parts","test"  ,u"1-3_関_品出し_uniken_step%d_processed_t.csv" % step)
#,"display",step)
#ini_parts_split(os.path.join(curdir,u"datas"
#,u"20161020_採取データ"                                 ,u"uniken_1_4_関_事務_胸_20161020172042.csv")
#,[os.path.join(curdir,u"datas",u"20161020_採取データ",u"uniken_1_4_関_事務_腕_20161020172041.csv")
#,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_4_関_事務_腰_20161020172040.csv")]
#,os.path.join(curdir,u"datas",u"concat_parts","learn" ,u"1-4_関_事務_uniken_step%d_processed_l.csv" % step)
#,os.path.join(curdir,u"datas",u"concat_parts","test"  ,u"1-4_関_事務_uniken_step%d_processed_t.csv" % step)
#,"officework",step)
#ini_parts_split(os.path.join(curdir,u"datas"
#,u"20161020_採取データ"                                 ,u"uniken_1_5_関_掃除_胸_20161020172749.csv")
#,[os.path.join(curdir,u"datas",u"20161020_採取データ",u"uniken_1_5_関_掃除_腕_20161020172747.csv")
#,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_5_関_掃除_腰_20161020172745.csv")]
#,os.path.join(curdir,u"datas",u"concat_parts","learn" ,u"1-5_関_掃除_uniken_step%d_processed_l.csv" % step)
#,os.path.join(curdir,u"datas",u"concat_parts","test"  ,u"1-5_関_掃除_uniken_step%d_processed_t.csv" % step)
#,"clean",step)
#ini_parts_split(os.path.join(curdir,u"datas"
#,u"20161020_採取データ"                                 ,u"uniken_1_7_関_歩行_胸_20161021121819.csv")
#,[os.path.join(curdir,u"datas",u"20161020_採取データ",u"uniken_1_7_関_歩行_腕_20161021121819.csv")
#,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_7_関_歩行_腰_20161021121819.csv")]
#,os.path.join(curdir,u"datas",u"concat_parts","learn" ,u"1-6_関_歩行_uniken_step%d_processed_l.csv" % step)
#,os.path.join(curdir,u"datas",u"concat_parts","test"  ,u"1-6_関_歩行_uniken_step%d_processed_t.csv" % step)
#,"walk",step)

####加工後ファイル読み込み###
#files =[ pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts" ,"learn" ,u"1-1_関_発注_uniken_step25_processed_l.csv"))
#       , pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts" ,"learn" ,u"1-2_関_棚卸_uniken_step25_processed_l.csv"))
#       , pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts" ,"learn" ,u"1-3_関_品出し_uniken_step25_processed_l.csv"))
#       , pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts" ,"learn" ,u"1-4_関_事務_uniken_step25_processed_l.csv"))
#        , pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts","learn" ,u"1-5_関_掃除_uniken_step25_processed_l.csv"))
#        , pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts","learn" ,u"1-6_関_歩行_uniken_step25_processed_l.csv"))
#         ]
###通常の学習＆テスト、グラフ表示###
#for algkey in algs.keys():
#    for num in range(25,26,5):
#        #加工済みデータから学習用データに加工実行
#        files = [
##                 pd.read_csv("datas\seki1\walking_step%d_processed.csv" % num)
##                ,pd.read_csv("datas\seki1\clotheshorse_step%d_processed.csv" % num)
##                ,pd.read_csv("datas\seki1\clotheshorse2_step%d_processed.csv" % num)
##                ,pd.read_csv("datas\seki1\walking2_step%d_processed.csv" % num)
##                ,pd.read_csv("datas\seki1\use_bus_step%d_processed.csv" % num)
##                ,pd.read_csv("datas\seki1\driving_step%d_processed.csv" % num)
##                ,
#                pd.read_csv("datas\seki2\carryheavyprinter_step%d_processed.csv" % num)
#                ,pd.read_csv("datas\seki2\carryheavyprinter2_step%d_processed.csv" % num)
#                ,pd.read_csv("datas\seki2\clotheshorse3_step%d_processed.csv" % num)
#                ,pd.read_csv("datas\seki2\drive2_step%d_processed.csv" % num)            
#                ,pd.read_csv("datas\seki2\printermente_step%d_processed.csv" % num)
##                ,pd.read_csv("datas\moriyama1\moriyama_walk1_step%d_processed.csv" % num)
##                , pd.read_csv("datas\moriyama1\moriyama_walk2_step%d_processed.csv" % num)
##                , pd.read_csv("datas\moriyama1\moriyama_driving_step%d_processed.csv" % num)
#                ]
#        
#        #y,X = proc_for_fit(f.loc[:,feuture_value_columns] for f in files)
#
#y,X = proc_for_fit(files)
##主成分分析
#import MyPCA as mpca
#myp = mpca.MyPCA(X, standard=True)
#X_pca = myp.pca_fit(n=40)
#print("統計量基準レコード数%d" % 25)
#for algkey in algs.keys():
#    myclf = run_ml(algs[algkey], X , y, 0.7, algkey, 
#               u"D:\Python\ML\learned_classifier\%s\%s" % (algkey,algkey),
#               use_saved_clf=False, show_plot=False
#               )

###交差検証###
        
        #y,X = proc_for_fit(f.loc[:,feuture_value_columns] for f in files)

#y,X = proc_for_fit(files)
#import MyPCA as mpca
#myp = mpca.MyPCA(X)
#X_pca = myp.pca_fit(n=70)
#print("交差検証　統計量基準レコード数%d" % 25)
#for algkey in algs.keys():
#    #交差検証実行
#    run_crossval(algs[algkey], X_pca , y, 0.3, algkey)
            

###学習用とは別の人のデータでテスト###
#files_learn =[ pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts",u"分割" ,"learn" ,u"1-1_関_発注_uniken_step25_processed_l.csv"))
#       , pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts" ,u"分割","learn" ,u"1-2_関_棚卸_uniken_step25_processed_l.csv"))
#       , pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts" ,u"分割","learn" ,u"1-3_関_品出し_uniken_step25_processed_l.csv"))
#       , pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts" ,u"分割","learn" ,u"1-4_関_事務_uniken_step25_processed_l.csv"))
#        , pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts",u"分割","learn" ,u"1-5_関_掃除_uniken_step25_processed_l.csv"))
#        , pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts",u"分割","learn" ,u"1-6_関_歩行_uniken_step25_processed_l.csv"))
##         ]
##files_test =[
#        ,pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts" ,u"分割", u"test" ,u"1-1_関_発注_uniken_step25_processed_t.csv"))
#       , pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts" ,u"分割",u"test" ,u"1-2_関_棚卸_uniken_step25_processed_t.csv"))
#       , pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts" ,u"分割",u"test" ,u"1-3_関_品出し_uniken_step25_processed_t.csv"))
#       , pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts" ,u"分割",u"test" ,u"1-4_関_事務_uniken_step25_processed_t.csv"))
#        , pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts",u"分割",u"test" ,u"1-5_関_掃除_uniken_step25_processed_t.csv"))
#        , pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts",u"分割",u"test" ,u"1-6_関_歩行_uniken_step25_processed_t.csv"))
#        ]

#files_test = [pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts",u"分割","test" ,u"1-7_関_事務荷物品出し発注_uniken_step25_processed_t.csv"))]
#files_test = [pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts",u"分割","test" ,u"1-7_関_事務荷物品出し発注_学習不可データなし_uniken_step25_processed_t.csv"))]
        
#y_learn,X_learn = proc_for_fit(files_learn)
#y_test,X_test = proc_for_fit(files_test)

#使う列をそろえる
#X_test = X_test.drop(["pitch","roll","azimuth"], axis=1)
      
##主成分分析
#import MyPCA as mpca
#myp1 = mpca.MyPCA(X_learn, standard=True)
#X_l_std = myp1.std
#X_l_pca = myp1.pca_fit(n=50)
#
#myp2 = mpca.MyPCA(X_test, standard=True)
#X_t_std = myp2.std
#X_t_pca = myp2.pca_fit(n=50)

#for algkey in algs.keys():
#    #学習用データで学習
##    model_inst = algs[algkey](X_learn,y_learn,0.01)
##    model_inst = algs[algkey](X_l_pca,y_learn,0.01)
#    model_inst = algs[algkey](X_l_std,y_learn,0.01)
#    model_inst.name = algkey
#    model_inst.fit()
#    
#    model_inst.show_score()
#    #テストデータを予測
##    model_inst.test_otherdata(y_test,X_test)
##    model_inst.test_otherdata(y_test,X_t_pca)
##    model_inst.test_otherdata(y_test,X_t_std)
#    pred = model_inst.predict_unknowndata(X_t_std)
#    #未知データテスト正解率表示
#    #ラベルを名称にエンコード
#    pred_enc = model_inst.class_la.inverse_transform(pred)
#    from sklearn.metrics import accuracy_score
#    print("%s 純テストデータ正答率 %.6f\r\n" % (algkey.encode('utf-8')
#            , accuracy_score(y_test, pred_enc)))
    #np.savetxt("predict_result.csv", pred_enc, delimiter=",",  fmt='%s')
    #y_test.to_csv("answer.csv", index=False, encoding='shift-jis')
    
###グリッドサーチ###
#X = pd.concat([X_learn,X_test],axis=0)
#y = pd.concat([y_learn,y_test],axis=0)
#step = 50
#y,X = proc_for_fit(
#            [pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", u"step_n",u"1-1_関_発注_uniken_step%d_processed.csv" % step)),
#             pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", u"step_n",u"1-2_関_棚卸_uniken_step%d_processed.csv" % step)),
#            pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", u"step_n",u"1-3_関_品出し_uniken_step%d_processed.csv" % step)),
#            pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", u"step_n",u"1-4_関_事務_uniken_step%d_processed.csv" % step)),
#            pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", u"step_n",u"1-5_関_掃除_uniken_step%d_processed.csv" % step)),
#            pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", u"step_n",u"1-7_関_歩行_uniken_step%d_processed.csv" % step)),
#            pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", u"step_n_test",u"1_6_関_事務荷物品出し発注_step%d_processed.csv" % step))]
#                               )
##for algkey in algs.keys():
#for algkey in [u"logistic_regression"]:
#    clf_inst = algs[algkey](X, y, 0.3)
#    clf_inst.name = algkey
#    clf_inst.grid_search(alg_params[algkey], cv_p=3)
    

##ナイーブベイズ使用（正規分布、ベルヌーイ分布、多項分布）
##負数は使えないので０から1にスケーリング
#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler()
#X_scl = sc.fit_transform(X.ix[:,["Xmss_Max","Ymss_Max"]])
#X_scl = sc.fit_transform(X)
#nb = md.naive_bayes(X_scl,y,0.3)
#nb.name = u"ナイーブベイズ"
#nbclfs = run_mls(nb,"D:/Python/ML/learned_classifier/naive_bayes/")
#for name,clf in nbclfs.items():
#    nb.name = name
#    showplot(nb, clf, 100, "x_max", "y_max")