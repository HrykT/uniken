# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 13:17:54 2016

@author: Tsukamoto
"""

import os
import MLProc as mlp
import pandas as pd
import codecs
import Single_Models as md

#共通変数・定数
curdir = os.getcwd()

####ステップ数別学習用ファイル加工####
def learn_data_proc():
    for step in (5,10,20,30,40,50,60,70,80,90,100,150,200,250,300):
        mlp.ini_parts(
                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_1_関_発注_胸_20161020165951.csv")
                 ,os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_1_関_発注_腕_20161020165950.csv")
                 ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_1_関_発注_腰_20161020165948.csv")
                 ,os.path.join(curdir,u"datas",u"concat_parts", u"step_n"  ,u"1-1_関_発注_uniken_step%d_processed.csv" % step)
                 ,"order",step)
    
        mlp.ini_parts(
                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_2_関_棚卸_胸_20161020170752.csv")
                 ,os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_2_関_棚卸_腕_20161020170749.csv")
                 ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_2_関_棚卸_腰_20161020170747.csv")
                 ,os.path.join(curdir,u"datas",u"concat_parts", u"step_n"  ,u"1-2_関_棚卸_uniken_step%d_processed.csv" % step)
                 ,"tanaoroshi",step)
    
        mlp.ini_parts(
                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_3_関_品出し_胸_20161020171357.csv")
                 ,os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_3_関_品出し_腕_20161020171356.csv")
                 ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_3_関_品出し_腰_20161020171354.csv")
                 ,os.path.join(curdir,u"datas",u"concat_parts", u"step_n"  ,u"1-3_関_品出し_uniken_step%d_processed.csv" % step)
                 ,"display",step)
    
        mlp.ini_parts(
                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_4_関_事務_胸_20161020172042.csv")
                 ,os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_4_関_事務_腕_20161020172041.csv")
                 ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_4_関_事務_腰_20161020172040.csv")
                 ,os.path.join(curdir,u"datas",u"concat_parts", u"step_n"  ,u"1-4_関_事務_uniken_step%d_processed.csv" % step)
                 ,"officework",step)
    
        mlp.ini_parts(
                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_5_関_掃除_胸_20161020172749.csv")
                 ,os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_5_関_掃除_腕_20161020172747.csv")
                 ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_5_関_掃除_腰_20161020172745.csv")
                 ,os.path.join(curdir,u"datas",u"concat_parts", u"step_n"  ,u"1-5_関_掃除_uniken_step%d_processed.csv" % step)
                 ,"cleaning",step)
    
        mlp.ini_parts(
                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_7_関_歩行_胸_20161021121819.csv")
                 ,os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_7_関_歩行_腕_20161021121819.csv")
                 ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_7_関_歩行_腰_20161021121819.csv")
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
    for step in (5,10,20,30,40,50,60,70,80,90,100,150,200,250,300):
        mlp.ini_parts(
                   os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_6_関_事務荷物品出し発注_胸_20161020173701.csv")
                 ,os.path.join(curdir,u"datas" ,u"20161020_採取データ"  ,u"uniken_1_6_関_事務荷物品出し発注_腕_20161020173659.csv")
                 ,os.path.join(curdir,u"datas",u"20161020_採取データ"  ,u"uniken_1_6_関_事務荷物品出し発注_腰_20161020173657.csv")
                 ,os.path.join(curdir,u"datas",u"concat_parts", u"step_n_test"  ,u"1_6_関_事務荷物品出し発注_step%d_processed.csv" % step)
                 ,"mix",step, mix_label=mixlabel)
        print(u"ステップ%d　処理終了" % step)

#実行
#test_data_proc()

######各種検証をループ#######
def print_and_outfile(out_file, text):
    print(text)
    #out_file.write(text)
    
def select_usevalue():
    #使用特徴量選択
    #センサー値
    val_sensor = mlp.stat_target
    #部位
    val_part = {u"胸腕腰":("_1","_2","_3"), u"腕腰":("_2","_3")}
    #統計
    val_stat = {u"平均・分散":("_Avg","_Var"),
                u"平均・分散・最大・最小":("_Avg","_Var", "_Max", "_Min")}
    
    use_val_se = []
    use_val_st = []
    use_vals = {}
    for kp,part_pattern in val_part.iteritems():
        #部位のパターン
        for ks,stat_pattern in val_stat.iteritems():
            #使用統計量のパターン
            use_val_se = []
            use_val_st = []
            for s in val_sensor:
                #センサー名ループ
                for p in part_pattern:
                    #センサー+部位
                    use_val_se.append(s + p)
                    use_vals[kp + u"／" + u"センサー値"] = use_val_se
                for p in part_pattern:
                    #センサーのみ、統計量のみを分けるため、部位のループをもう一度やる
                    for v in stat_pattern:
                        use_val_st.append(s + v + p)
                        use_vals[kp + u"／" + ks + u"／" + u"統計値"] = use_val_st
            use_vals[kp + u"／" + ks + u"／" + u"全て"] = use_val_se + use_val_st
    return use_vals

def run_conbinationtest(out_file):
    from sklearn.preprocessing import StandardScaler
    std = StandardScaler()
    
    #使用する特徴量のパターンをリストで持つ
    feuture_values = select_usevalue()
    #標準出力表示用のインデント
    indent = "    "
    
    res_list = pd.DataFrame()
    for step in [100]:
    #for step in (30,40,50,60,70,80,90,100,150,200,250,300):
    #for step in (5,10,20,30,40,50,60,70,80,90,100,150,200,250,300):
        print_and_outfile(out_file, u"ステップ数%d" % step)
        #統計量取得ステップごとのループ
        #学習データ読込
        y_learn,X_learn = mlp.proc_for_fit(
            [pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", u"step_n",u"1-1_関_発注_uniken_step%d_processed.csv" % step)),
             pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", u"step_n",u"1-2_関_棚卸_uniken_step%d_processed.csv" % step)),
            pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", u"step_n",u"1-3_関_品出し_uniken_step%d_processed.csv" % step)),
            pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", u"step_n",u"1-4_関_事務_uniken_step%d_processed.csv" % step)),
            pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", u"step_n",u"1-5_関_掃除_uniken_step%d_processed.csv" % step)),
            pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", u"step_n",u"1-7_関_歩行_uniken_step%d_processed.csv" % step))]
                               )
        #テスト対象データ読込
        test = pd.read_csv(os.path.join(curdir,u"datas",u"concat_parts", u"step_n_test",u"1_6_関_事務荷物品出し発注_step%d_processed.csv" %step))
    
        #テスト対象から学習不可行動を除く
        filter_notlearnlbel = (test['label'] != 'nimotsu') & \
                                (test['label'] != 'service') 
        y_test,X_test = mlp.proc_for_fit([test[filter_notlearnlbel]])
        
        for std_fit in [True]:
#        for std_fit in (True,False):
            #正規化有無のループ
            print_and_outfile(out_file, indent * 1 + u"正規化：%s" % str(std_fit))
            #for fk in feuture_values.iterkeys():
            for fk in [u"胸腕腰／平均・分散・最大・最小／全て"]:
                #使用特徴量ごとのループ
                print_and_outfile(out_file, indent * 2 + fk)
                feuture_value = feuture_values[fk]
                #使用する特徴量抽出
                X_learn_limit = X_learn[feuture_value]
                X_test_limit = X_test[feuture_value]
                
                if std_fit:
                    #データ選択し終えたら正規化
                    X_learn_limit = std.fit_transform(X_learn_limit)
                    X_test_limit = std.fit_transform(X_test_limit)
                
                #結果行
                res_list = pd.DataFrame()
                res_row = pd.DataFrame([[0] * len(mlp.algs.keys())])
                res_row.columns = mlp.algs.keys()
                #for algkey in mlp.algs.keys():
                for algkey in [u"Adaboost"]:
                    #使用するアルゴリズムごとのループ
                    #推定器の学習、推定
                    model_inst = mlp.algs[algkey](X_learn_limit,y_learn,0.01)
                    model_inst.name = algkey
                    model_inst.fit()
                    pred = model_inst.predict_unknowndata(X_test_limit)
                    #pred_enc = model_inst.class_la.inverse_transform(pred)
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

def ensemble_Voting():
    from sklearn.ensemble import VotingClassifier
    step = 100
    
