# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 13:17:54 2016

@author: Tsukamoto
"""

import os
import pandas as pd
import codecs
import numpy as np
import MLProc as mlp
import Single_Models as md
import InitDataProcess as ini

#共通変数・定数
curdir = os.getcwd()

####ステップ数別学習用ファイル加工####
def learn_data_proc(data_proc, outfld):
#    for step in (5,10,20,30,40,50,60,70,80,90,100,150,200,250,300):
    for step in [60,100]:
#        data_proc(
#                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_1_関_発注_胸_20161020165951.csv")
#                 ,os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_1_関_発注_腕_20161020165950.csv")
#                 ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_1_関_発注_腰_20161020165948.csv")
#                 ,os.path.join(curdir,u"datas",u"concat_parts",  outfld, "20161020"  ,u"1-1_関_発注_uniken_step%d_processed.csv" % step)
#                 ,"order",step)
#    
#        data_proc(
#                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_2_関_棚卸_胸_20161020170752.csv")
#                 ,os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_2_関_棚卸_腕_20161020170749.csv")
#                 ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_2_関_棚卸_腰_20161020170747.csv")
#                 ,os.path.join(curdir,u"datas",u"concat_parts", outfld , "20161020" ,u"1-2_関_棚卸_uniken_step%d_processed.csv" % step)
#                 ,"tanaoroshi",step)
#    
#        data_proc(
#                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_3_関_品出し_胸_20161020171357.csv")
#                 ,os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_3_関_品出し_腕_20161020171356.csv")
#                 ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_3_関_品出し_腰_20161020171354.csv")
#                 ,os.path.join(curdir,u"datas",u"concat_parts", outfld , "20161020" ,u"1-3_関_品出し_uniken_step%d_processed.csv" % step)
#                 ,"display",step)
#    
#        data_proc(
#                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_4_関_事務_胸_20161020172042.csv")
#                 ,os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_4_関_事務_腕_20161020172041.csv")
#                 ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_4_関_事務_腰_20161020172040.csv")
#                 ,os.path.join(curdir,u"datas",u"concat_parts", outfld , "20161020" ,u"1-4_関_事務_uniken_step%d_processed.csv" % step)
#                 ,"officework",step)
#    
#        data_proc(
#                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_5_関_掃除_胸_20161020172749.csv")
#                 ,os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_5_関_掃除_腕_20161020172747.csv")
#                 ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_5_関_掃除_腰_20161020172745.csv")
#                 ,os.path.join(curdir,u"datas",u"concat_parts", outfld , "20161020" ,u"1-5_関_掃除_uniken_step%d_processed.csv" % step)
#                 ,"cleaning",step)
#    
#        data_proc(
#                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_7_関_歩行_胸_20161021121819.csv")
#                 ,os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_7_関_歩行_腕_20161021121819.csv")
#                 ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_7_関_歩行_腰_20161021121819.csv")
#                 ,os.path.join(curdir,u"datas",u"concat_parts", outfld , "20161020" ,u"1-7_関_歩行_uniken_step%d_processed.csv" % step)
#                 ,"walk",step)
#        data_proc(
#                   os.path.join(curdir,u"datas",u"20161102_採取データ",u"uniken_1_1_関_発注_胸_20161102155324.csv")
#                 ,os.path.join(curdir,u"datas" ,u"20161102_採取データ",u"uniken_1_1_関_発注_腕_20161102155324.csv")
#                 ,os.path.join(curdir,u"datas" ,u"20161102_採取データ",u"uniken_1_1_関_発注_腰_20161102155324.csv")
#                 ,os.path.join(curdir,u"datas" ,u"concat_parts", outfld , "20161102" ,u"1-1_関_発注_uniken_step%d_processed.csv" % step)
#                 ,"order",step)
#
#        data_proc(
#                   os.path.join(curdir,u"datas",u"20161102_採取データ",u"uniken_1_2_関_モップがけ_胸_20161102160721.csv")
#                 ,os.path.join(curdir,u"datas" ,u"20161102_採取データ",u"uniken_1_2_関_モップがけ_腕_20161102160721.csv")
#                 ,os.path.join(curdir,u"datas" ,u"20161102_採取データ",u"uniken_1_2_関_モップがけ_腰_20161102160721.csv")
#                 ,os.path.join(curdir,u"datas" ,u"concat_parts", outfld , "20161102" ,u"1-2_関_モップ_uniken_step%d_processed.csv" % step)
#                 ,"cleaning",step)
#
#        data_proc(
#                   os.path.join(curdir,u"datas",u"20161102_採取データ",u"uniken_1_3_関_棚卸_胸_20161102161332.csv")
#                 ,os.path.join(curdir,u"datas" ,u"20161102_採取データ",u"uniken_1_3_関_棚卸_腕_20161102161332.csv")
#                 ,os.path.join(curdir,u"datas" ,u"20161102_採取データ",u"uniken_1_3_関_棚卸_腰_20161102161332.csv")
#                 ,os.path.join(curdir,u"datas" ,u"concat_parts", outfld , "20161102" ,u"1-3_関_棚卸_uniken_step%d_processed.csv" % step)
#                 ,"tanaoroshi",step)
#
#        data_proc(
#                   os.path.join(curdir,u"datas",u"20161102_採取データ",u"uniken_1_4_関_パソコン事務_胸_20161102174818.csv")
#                 ,os.path.join(curdir,u"datas" ,u"20161102_採取データ",u"uniken_1_4_関_パソコン事務_腕_20161102174818.csv")
#                 ,os.path.join(curdir,u"datas" ,u"20161102_採取データ",u"uniken_1_4_関_パソコン事務_腰_20161102174818.csv")
#                 ,os.path.join(curdir,u"datas" ,u"concat_parts", outfld , "20161102" ,u"1-4_関_事務_uniken_step%d_processed.csv" % step)
#                 ,"officework",step)

        print(u"ステップ%d　処理終了" % step)
#実行
#learn_data_proc(mlp.ini_parts ,u"step_n")


####ステップ数別予測用ファイル加工####
def test_data_proc(data_proc, outfld):
    import pandas as pd
#    mixlabel = pd.DataFrame([
#    ["2016/10/20 17:36:58",
#     "2016/10/20 17:38:50",
#    "2016/10/20 17:39:18",
#    "2016/10/20 17:40:36",
#    "2016/10/20 17:41:04",
#    "2016/10/20 17:42:25",
#    "2016/10/20 17:42:56",
#    "2016/10/20 17:43:38"],
#    ["officework",
#     "walk",
#     "nimotsu",
#     "walk",
#     "display",
#     "service",
#     "walk",
#     "order"]
#    ]).T

    mixlabel = pd.DataFrame([
    [
    "2016/12/02 16:22:56",
    "2016/12/02 17:21:20",
    "2016/12/02 17:23:40",
    "2016/12/02 17:26:10",
    "2016/12/02 17:28:15",
    "2016/12/02 17:32:40",
    "2016/12/02 17:35:10",
    "2016/12/02 17:37:50",
    ],
    [u"事務作業",
     u"箱を置く",
     u"カート移動",
     u"ＰＯＰ付け外し",
     u"前出し",
     u"陳列",
     u"発注",
     u"配置変更",
     ]
    ]).T
                              
#    for step in (5,10,20,30,40,50,60,70,80,90,100,150,200,250,300):
    for step in [60,100]:
#        data_proc(
#                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_6_関_事務荷物品出し発注_胸_20161020173701.csv")
#                 ,os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_6_関_事務荷物品出し発注_腕_20161020173659.csv")
#                 ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_6_関_事務荷物品出し発注_腰_20161020173657.csv")
#                 ,os.path.join(curdir,u"datas",u"concat_parts", outfld  ,u"1_6_関_事務荷物品出し発注_step%d_processed.csv" % step)
#                 ,"mix",step, mix_label=mixlabel)
        data_proc(
                   os.path.join(curdir,u"datas",u"20161202_採取データ",u"uniken_胸_20161202162346.csv")
                 ,os.path.join(curdir,u"datas" ,u"20161202_採取データ",u"uniken_腰_20161202162314.csv")
                 ,os.path.join(curdir,u"datas" ,u"20161202_採取データ",u"uniken_腕_20161202162256.csv")
                 ,os.path.join(curdir,u"datas" ,u"concat_parts", outfld , "20161202" ,u"uniken_20161202_step%d_processed.csv" % step)
                 ,"mix",step, mix_label=mixlabel)
        print(u"ステップ%d　処理終了" % step)

#実行
#test_data_proc(mlp.ini_parts,u"step_n")

def labellogger_concat_ini(data_proc, outfld):
    #ラベルデータ
    labels1 = pd.read_csv(os.path.join(curdir, u"datas" ,u"20161114_実地採取データ", u"20161114_女池店小林様.csv")
                       , sep=",", encoding='shift-jis')
    labels2 = pd.read_csv(os.path.join(curdir, u"datas" ,u"20161128_実地採取データ", u"20161128_ラベル.csv")
                       , sep=",", encoding='shift-jis')
    labels1 = labels1.ix[:,[0,3]]
    
    labels1.columns = ['datetime','label']
    labels2.columns = ['datetime','place','label','arm','move']

    for step in [60,100]:
        data_proc(os.path.join(curdir,u"datas" ,u"20161114_実地採取データ"  ,u"uniken_20161114_071744_breast_correction.csv")
                 ,os.path.join(curdir,u"datas" ,u"20161114_実地採取データ"  ,u"uniken_腕_20161114071435.csv")
                 ,os.path.join(curdir,u"datas",u"20161114_実地採取データ"  ,u"uniken_腰_20161114.csv")
                 ,os.path.join(curdir,u"datas",u"concat_parts",  outfld, "20161114"  ,u"uniken_20161114_step%d.csv" % step)
                 ,"",step, label_data=labels1)
        data_proc(os.path.join(curdir,u"datas" ,u"20161128_実地採取データ"  ,u"uniken_腕_20161128.csv")
                 ,os.path.join(curdir,u"datas",u"20161128_実地採取データ"  ,u"uniken_腰_20161128.csv")
                 ,""
                 ,os.path.join(curdir,u"datas",u"concat_parts",  outfld, "20161128"  ,u"uniken_20161128_step%d.csv" % step)
                 ,"",step, label_data=labels2)
    print(u"ステップ%d　処理終了" % step)

#labellogger_concat_ini(mlp.ini_parts, u"step_n")
    
######各種検証をループ#######
def print_and_outfile(out_file, text):
    print(text)
    #out_file.write(text)

def get_data(step, fold, use_clm="腕腰／平均・分散・最大・最小／全て"):
    #ラベルなし、その他ラベルは除く
    data = pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", fold, u"20161128",u"uniken_20161128_step%d.csv" % step)
                        ,encoding='shift-jis').dropna(axis=0, subset=['label'])
#    data = pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", fold, u"20161114",u"uniken_20161114_step%d.csv" % step)
#                        ,encoding='shift-jis').dropna(axis=0, subset=['label'])
    without_other = data['label'] != u'その他'

    y_learn,X_learn = mlp.proc_for_fit(
    [
#     pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", fold, u"20161020",u"1-1_関_発注_uniken_step%d_processed.csv" % step)),
#     pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", fold, u"20161020",u"1-2_関_棚卸_uniken_step%d_processed.csv" % step)),
#     pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", fold, u"20161020",u"1-3_関_品出し_uniken_step%d_processed.csv" % step)),
#     pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", fold, u"20161020",u"1-4_関_事務_uniken_step%d_processed.csv" % step)),
#     pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", fold, u"20161020",u"1-5_関_掃除_uniken_step%d_processed.csv" % step)),
#     pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", fold, u"20161020",u"1-7_関_歩行_uniken_step%d_processed.csv" % step)),
#     pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", fold, u"20161102",u"1-1_関_発注_uniken_step%d_processed.csv" % step)),
#     pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", fold, u"20161102",u"1-2_関_モップ_uniken_step%d_processed.csv" % step)),
#     pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", fold, u"20161102",u"1-3_関_棚卸_uniken_step%d_processed.csv" % step)),
#     pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", fold, u"20161102",u"1-4_関_事務_uniken_step%d_processed.csv" % step)),
      #data[without_other].drop(['place','move','arm'], axis=1),
      data[without_other],
#      pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", fold, u"20161202",u"uniken_20161202_step%d_filterd.csv" % step)),
     ]
     )
    #テスト対象データ読込
    test = pd.concat([
#      data[without_other],
        pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", "step_n_test", u"1_6_関_事務荷物品出し発注_step%d_processed.csv" % step)),
     ],ignore_index=True)
   
    #テスト対象から学習不可行動を除く
    filter_notlearnlbel = (test['label'] != 'nimotsu') & \
                            (test['label'] != 'service') 
    y_test,X_test = mlp.proc_for_fit([test[filter_notlearnlbel]])
    
    from sklearn.preprocessing import StandardScaler
    std = StandardScaler()
    
    feuture_values = select_usevalue(feuture_type='stat')
    feuture_value = feuture_values[use_clm]
    xl_std = std.fit_transform(X_learn[feuture_value])
    xt_std = std.fit_transform(X_test[feuture_value])
    
    return y_learn,xl_std,y_test,xt_std

def get_pred_target_data(path, use_clm="腕腰／平均・分散・最大・最小／全て"):
    #予測だけ行うデータの加工
    data = pd.read_csv(path)
    time = data.loc[:,"datetime"]
    
    feuture_values = select_usevalue(feuture_type='stat')
    feuture_value = feuture_values[use_clm]
    pred_data = data.loc[:,feuture_value]
    
    return time, pred_data
    
def select_usevalue(feuture_type='stat'):
    
    use_vals = {}
    #センサー値
    val_sensor = mlp.stat_target
    #部位
    val_part = {u"胸腕腰":("_1","_2","_3"), u"腕腰":("_2","_3")}
    #val_part = {u"胸腕腰":("_1","_2","_3"), u"腕腰":("_1","_2")}
                
    if feuture_type == 'stat':
        #使用特徴量選択のパターンを生成し、文字列辞書で持つ
        #キー：パターンを説明する日本語　値：使用する列名の配列
        #統計
        val_stat = {u"平均・分散":("_Avg","_Var"),
                    u"平均・分散・最大・最小":("_Avg","_Var", "_Max", "_Min")}
        
        use_val_se = []
        use_val_st = []
        for kp,part_pattern in val_part.iteritems():
            #部位のパターン
            for ks,stat_pattern in val_stat.iteritems():
                #使用統計量のパターン
                use_val_se = []
                use_val_st = []
                for s in val_sensor:
                    #センサー名ループ
    #                for p in part_pattern:
    #                    #センサー+部位
    #                    use_val_se.append(s + p)
    #                    use_vals[kp + u"／" + u"センサー値"] = use_val_se
                    for p in part_pattern:
                        #センサーのみ、統計量のみを分けるため、部位のループをもう一度やる
                        for v in stat_pattern:
                            use_val_st.append(s + v + p)
                            use_vals[kp + u"／" + ks + u"／" + u"統計値"] = use_val_st
                use_vals[kp + u"／" + ks + u"／" + u"全て"] = use_val_se + use_val_st
    elif feuture_type == 'next':
        #次レコード行を列として追加する処理で、何行先までの特徴量を使うか選択
        #n行先
        val_step = {u"":u"追加なし", u"_1":u"1行先", u"_2":u"2行先", u"_3":u"3行先", u"_4":u"4行先"}
        for kp,part_pattern in val_part.iteritems():
            #部位のパターン
            use_val_se = []
            for p in part_pattern:
                for kst,vst in val_step.iteritems():
                    for s in val_sensor:
                        use_val_se.append(s + p + kst)
                    use_vals[kp + u"／" + vst] = use_val_se
    return use_vals
    
def run_conbinationtest(out_file):
    from sklearn.preprocessing import StandardScaler
    std = StandardScaler()
    
    #使用する特徴量のパターンをリストで持つ
    #feuture_values = select_usevalue(feuture_type='stat')

    #標準出力表示用のインデント
    indent = "    "
    
    res_list = pd.DataFrame()
    for step in [100]:
    #for step in (30,40,50,60,70,80,90,100,150,200,250,300):
    #for step in (5,10,20,30,40,50,60,70,80,90,100,150,200,250,300):
        print_and_outfile(out_file, u"ステップ数%d" % step)
        #統計量取得ステップごとのループ
        #学習データ、テストデータ読込
        y_learn, X_learn, y_test, X_test = get_data(step, "step_n")
        
        for std_fit in [True]:
        #for std_fit in (True,False):
            #正規化有無のループ
            print_and_outfile(out_file, indent * 1 + u"正規化：%s" % str(std_fit))
            for fk in feuture_values.iterkeys():
            #for fk in [u"腕腰／平均・分散・最大・最小／全て"]:
                #使用特徴量ごとのループ
                print_and_outfile(out_file, indent * 2 + fk)
                feuture_value = feuture_values[fk]
                #使用する特徴量抽出
                X_learn_limit = X_learn[feuture_value]
                X_test_limit = X_test[feuture_value]
                
#                if std_fit:
#                    #データ選択し終えたら正規化
#                    X_learn_limit = std.fit_transform(X_learn_limit)
#                    X_test_limit = std.fit_transform(X_test_limit)
                
                #結果行
                res_list = pd.DataFrame()
                res_row = pd.DataFrame([[0] * len(mlp.algs.keys())])
                res_row.columns = mlp.algs.keys()
                #for algkey in mlp.algs.keys():
                for algkey in [u"logistic_regression"]:
                    #使用するアルゴリズムごとのループ
                    #推定器の学習、推定
                    model_inst = mlp.algs[algkey](X_learn_limit,y_learn,0.01,C=1.0)
                    model_inst.name = algkey
                    model_inst.fit()
                    pred = model_inst.predict_unknowndata(X_test_limit)
                    #pred_enc = model_inst.class_la.inverse_transform(pred)
                    np.savetxt("answer.csv", y_test, delimiter=",", fmt="%s")
                    np.savetxt("predict_result.csv", model_inst.class_la.inverse_transform(pred), delimiter=",", fmt="%s")
                    y_test_enc = model_inst.class_la.transform(y_test)
                    from sklearn.metrics import accuracy_score
                    scr = accuracy_score(y_test_enc, pred)
                    res_row.loc[:,algkey] = scr

    #                print(indent * 2 + "%s,%.6f\r\n" % (algkey.encode('utf-8')
    #                       , scr))
                res_list = res_list.append(res_row)
                print_and_outfile(out_file, res_list)
 

#実行
#res_file = codecs.open(u'MLParamSearchResult.html', 'a', 'utf-8')
#try:
#    run_conbinationtest(res_file)
#finally:
#    res_file.close()

def ensemble_voting(save=False):
    #アンサンブル学習
    import Voting_Models as vm
    import itertools

    step = 100
    yl,xl,yt,xt=get_data(step,"step_n")
    
    #使えそうなアルゴリズムの一覧
    voter = {
             #u"svm_linear", u"svm_rbf",
             u"ｋ_neighbors" : md.kNeighbors(xl,yl,0.2, n=5, weights='uniform'), 
             #u"neural_net",
             u"logistic_regression" : md.logistic_regression(xl,yl,0.2, C=100),
             #u"RandomForest",
             u"GBDT" : md.GBDT(xl,yl,0.2, max_depth=1,
                                                     n_estimators=100
                                                     ,learning_rate=0.1)
             }
    conb = list(itertools.combinations(voter.keys(),3));
        
    for vc in conb:
        clfs = []
        #使う推定機のリストを作る
        for k in vc:
            clfs.append((k,voter[k].clf))
        #アンサンブル推定機の学習
        ens = vm.Voting_Model(xl, yl, 0.01, clfs)
        ens.name = u"Voting[%s]" % ",".join(vc)
        ens.fit()
        
        #学習データ内での精度
        ens.show_score()        
        
        #精度
        ens.test_otherdata(yt, xt)
        
        #保存
        if save:
            from datetime import datetime as dt
            s = md.modelIO()
            s.save_model(ens.clf
                , os.path.join("clfs","ens_%s_%s" % 
                               ("_".join(vc), dt.now().strftime("%Y%m%d%H%M%S"))))
            #ラベルの変換リストを残す
            ln = np.unique(yl)
            lnum = ens.class_la.transform(ln)
            label_list = (np.c_[ln,lnum]).T
            np.savetxt(os.path.join("clfs","label_%s_%s" % 
                               ("_".join(vc), dt.now().strftime("%Y%m%d%H%M%S")))
                ,label_list, fmt="%s", delimiter=",")
    
#ensemble_voting(save=True)
    
def closs_val():
    step = 60
    yl,xl,yt,xt=get_data(step, "step_n")
    pd.DataFrame(yl).to_csv('label_list.txt', index=False, encoding='shift-jis')
#    #for algkey in mlp.algs.keys():
#    for algkey in [u"svm_rbf",u"logistic_regression",u"ｋ_neighbors"]:
#        #交差検証実行
#        clf = mlp.algs[algkey](xl , yl, 0.1)
#        clf.name = algkey
#        clf.closs_vld(k=3, sampling_type='stkfold')
#closs_val()

def gridsearch():    
    step = 60
    yl,xl,yt,xt=get_data(step,"step_n")
    
    for algkey in [u"ｋ_neighbors",u"svm_rbf",u"logistic_regression"]:
#    for algkey in mlp.algs.keys():
        #使用するアルゴリズムごとのループ
        #推定器の学習、推定
        model_inst = mlp.algs[algkey](xl,yl,0.3)
        model_inst.name = algkey
        #model_inst.my_grid_search(mlp.alg_params[algkey], yt, xt)
        model_inst.my_grid_search(mlp.alg_params[algkey], yt, xt)
#gridsearch()

def learning():
    #学習のみ行って推定器を保存
    step = 60
    date = u"20161128"
#    p = [u"胸腕腰", u"腕腰"]
    p = [u"腕腰"]

    for parts in p:
        yl,xl,yt,xt=get_data(step, "step_n", use_clm= parts + u"／平均・分散・最大・最小／全て")
        
        #ロジスティック回帰
        path = os.path.join(curdir,u"learned_classifier"  ,u"logistic_regression",date,parts)
        lg = md.logistic_regression(xl,yl,0.1, C=1)
        #ラベル・エンコード値保存
        lbl = pd.DataFrame()
        lbl.loc[:,"label"] = yl
        lbl.loc[:,"num"] = lg.enc_target
        lbl.to_csv(os.path.join(path,u"label_lgr.csv") ,index=False, encoding='shift-jis')
#        lbl = np.c_[yl, lg.enc_target]
#        np.savetxt(os.path.join(path,u"label_lgr.csv") ,yl, fmt="%s")
        #学習、推定器保存
        clf = lg.fit_all()
        sv = md.modelIO()
        sv.save_model(clf,os.path.join(path,"lgr_" + date))
    
        #k近傍
        path = os.path.join(curdir,u"learned_classifier"  ,u"k_neighbors",date,parts)
        kn = md.kNeighbors(xl,yl,0.1,n=5,weights='uniform')
        #ラベル・エンコード値保存
        lbl = pd.DataFrame()
        lbl.loc[:,"label"] = yl
        lbl.loc[:,"num"] = kn.enc_target
        lbl.to_csv(os.path.join(path,u"label_kn.csv") ,index=False, encoding='shift-jis')
#        lbl = np.c_[yl, kn.enc_target]
#        np.savetxt(os.path.join(path,"label_kn.csv") ,lbl, fmt="%s")
        #学習、推定器保存
        clf = kn.fit_all()
        sv = md.modelIO()
        sv.save_model(clf,os.path.join(path,"kn_" + date))
        
        #SVM(RBF)
        path = os.path.join(curdir,u"learned_classifier", u"svm_rbf",date, parts)
        rbf = md.svm_rbf(xl,yl,0.1,C=10,gamma=0.001)
        #ラベル・エンコード値保存
        lbl = pd.DataFrame()
        lbl.loc[:,"label"] = yl
        lbl.loc[:,"num"] = rbf.enc_target
        lbl.to_csv(os.path.join(path,u"label_svm_rbf.csv") ,index=False, encoding='shift-jis')
#        lbl = np.c_[yl, rbf.enc_target]
#        np.savetxt(os.path.join(path,"label_svm_rbf.csv") ,lbl, fmt="%s")
        #学習、推定器保存
        clf = rbf.fit_all()
        sv = md.modelIO()
        sv.save_model(clf,os.path.join(path,"sbm_rbf_" + date))
    
#learning()
    
def Unsupervise_to_supervise():
    step = 100
    yl,xl,yt,xt=get_data(step,"step_n")
    data = np.r_[xl,xt]
    true_label = np.r_[yl,yt]
    
#    #教師なし学習でのクラス分け
#    clst = md.k_means(n_clusters=8)
#    #clst = md.AgglomerativeClustering()
#    labels = pd.DataFrame(clst.clustering(data))
#
#    #1-of-K表現にする
#    from sklearn.preprocessing import OneHotEncoder
#    ohe = OneHotEncoder(categorical_features=[0])
#    class_data = ohe.fit_transform(labels).toarray()
#    
#    data_pd = pd.DataFrame(class_data)
#    class_clm = ["0","1","2","3","4","5","6","7"]
#    data_pd.columns = class_clm
#
#    #分けたクラスを新たな学習データとして生成（５秒先までのクラスラベルを列に持つ）
#    new_data = class_data.copy()
#    for step in range(1,5):
#        #先頭step行を削除したデータを作る
#        base_del = class_data[step-1:]
#        #元データと結合する
#        new_data = np.c_[new_data[:len(base_del)], base_del]
    
    #統計量を追加する
#    ipr = ini.InitDataProc(data_pd, class_clm, "", n=50)
#    ipr.add_statvals(mlp.name_func)
#    ipr.output("movelabel_with_stat.csv")
    new_data = pd.read_csv("movelabel_with_stat_20.csv", 
                           sep=",", encoding='shift-jis')
                                   
    #保存
    #np.savetxt("move_lavel.csv",new_data, delimiter=",", fmt="%s")
    
    #正解ラベル切り出し
    true_label = true_label[:len(new_data)]
                            
    #元データに結合
    #new_data = np.c_[new_data, data[:len(new_data)]]
    
    #いくつか教師あり分類器に学習させる
    clf = md.svm_rbf(new_data[:20000], pd.DataFrame(true_label[:20000]),0.2)
    clf.fit()
    clf.show_score()
    
    #学習済み分類器で予測
    #pred = clf.clf.predict(new_data[20000:])
    
    #精度
    clf.my_grid_search({'n_estimators' : [100], 'max_depth' : [1,3,5],
                      'learning_rate' : [0.1, 0.2, 0.3]},
                         true_label[20000:], new_data[20000:])
    
#Unsupervise_to_supervise()

def without_walklabel():
    step = 100
    yl,xl,yt,xt=get_data(step,"step_n")
    data = pd.DataFrame(np.r_[xl,xt])
    true_label = pd.DataFrame(np.r_[yl,yt],columns=['label'])

    #walkを排除
    without_walk = true_label['label'] != 'walk'
    
    true_label = true_label[without_walk]
    data = data[without_walk]

    #いくつか教師あり分類器に学習させる
    clf = md.svm_rbf(data[:15000], pd.DataFrame(true_label[:15000]),0.2)
    clf.fit()
    clf.show_score()
    
    clf.my_grid_search( {'C': [10, 100], 'kernel': ['rbf']
                         , 'gamma': [0.001, 0.1, 1.0]},
                         true_label[15000:], data[15000:])    

#without_walklabel()
    
def out_fb_result(clf_path, data_path):
    #保存した推定器でデータを予測し結果を保存　フィードバック用の結果も作成する
    from datetime import datetime as dt
    l = md.modelIO()
    
    clf = l.load_model(clf_path)
    time, data = get_pred_target_data(data_path, u"腕腰／平均・分散・最大・最小／全て")
    #予測実行、pandasデータフレームとして取得
    pred = pd.DataFrame(clf.predict(data), columns=["result"])
    
    #ラベル名に変換して時間データと結合
    #label_list = pd.read_csv(label_encode_path)
    #pred = pred.replace(label_list)
    
    res = pd.concat([time,pred], axis=1)
    
    res.to_csv(os.path.join(u"results"
                   ,u"origin_result%s.csv" % dt.now().strftime("%Y%m%d%H%M%S"))
                   ,index=False)

#11/14データを学習した推定器で11/28を予測
#out_fb_result(os.path.join(curdir,u"learned_classifier",u"logistic_regression",u"20161114",u"腕腰",u"lgr_20161114"),
#              os.path.join(curdir,"datas",u"concat_parts",u"step_n","20161128",u"uniken_20161128_step60.csv")
#              )
#out_fb_result(os.path.join(curdir,u"learned_classifier",u"k_neighbors",u"20161114",u"腕腰",u"kn_20161114"),
#              os.path.join(curdir,"datas",u"concat_parts",u"step_n","20161128",u"uniken_20161128_step60.csv")
#              )
#out_fb_result(os.path.join(curdir,u"learned_classifier",u"svm_rbf",u"20161114",u"腕腰",u"sbm_rbf_20161114"),
#              os.path.join(curdir,"datas",u"concat_parts",u"step_n","20161128",u"uniken_20161128_step60.csv")
#              )

#out_fb_result(os.path.join(curdir,u"learned_classifier",u"logistic_regression",u"20161128",u"腕腰",u"lgr_20161128"),
#              os.path.join(curdir,"datas",u"concat_parts",u"step_n","20161114",u"uniken_20161114_step60.csv")
#              )
#out_fb_result(os.path.join(curdir,u"learned_classifier",u"k_neighbors",u"20161128",u"腕腰",u"kn_20161128"),
#              os.path.join(curdir,"datas",u"concat_parts",u"step_n","20161114",u"uniken_20161114_step60.csv")
#              )
#out_fb_result(os.path.join(curdir,u"learned_classifier",u"svm_rbf",u"20161128",u"腕腰",u"sbm_rbf_20161128"),
#              os.path.join(curdir,"datas",u"concat_parts",u"step_n","20161114",u"uniken_20161114_step60.csv")
#              )

#out_fb_result(os.path.join(curdir,u"learned_classifier",u"logistic_regression",u"20161202",u"胸腕腰",u"lgr_20161202"),
#              os.path.join(curdir,"datas",u"concat_parts",u"step_n","20161114",u"uniken_20161114_step60.csv")
#              )
#out_fb_result(os.path.join(curdir,u"learned_classifier",u"k_neighbors",u"20161202",u"胸腕腰",u"kn_20161202"),
#              os.path.join(curdir,"datas",u"concat_parts",u"step_n","20161114",u"uniken_20161114_step60.csv")
#              )
#out_fb_result(os.path.join(curdir,u"learned_classifier",u"svm_rbf",u"20161202",u"胸腕腰",u"sbm_rbf_20161202"),
#              os.path.join(curdir,"datas",u"concat_parts",u"step_n","20161114",u"uniken_20161114_step60.csv")
#              )
#
#out_fb_result(os.path.join(curdir,u"learned_classifier",u"logistic_regression",u"20161202",u"腕腰",u"lgr_20161202"),
#              os.path.join(curdir,"datas",u"concat_parts",u"step_n","20161128",u"uniken_20161128_step60.csv")
#              )
#out_fb_result(os.path.join(curdir,u"learned_classifier",u"k_neighbors",u"20161202",u"腕腰",u"kn_20161202"),
#              os.path.join(curdir,"datas",u"concat_parts",u"step_n","20161128",u"uniken_20161128_step60.csv")
#              )
#out_fb_result(os.path.join(curdir,u"learned_classifier",u"svm_rbf",u"20161202",u"腕腰",u"sbm_rbf_20161202"),
#              os.path.join(curdir,"datas",u"concat_parts",u"step_n","20161128",u"uniken_20161128_step60.csv")
#              )