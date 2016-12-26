# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 17:49:15 2016

@author: tsukamoto
"""
import pandas as pd
import numpy as np

class modelIO:
    """学習済みモデルの保存・読込"""
    def __init__(self):
        self
    def save_model(self,model,path):
        from sklearn.externals import joblib
        joblib.dump(model, path)
    def load_model(self,path):
        from sklearn.externals import joblib
        return joblib.load(path)

class single_model_base:
    """単体の分類器を動かす用のベースクラス"""
    def __init__(self,base,target,test_rate):
        #分類期の名前
        self.name = "nameless"
        #初期化した分類器
        self.clf = None
        #base:特徴量
        #target:正解ラベル
        from sklearn.preprocessing import LabelEncoder
        #ラベルを整数値にエンコード
        self.class_la = LabelEncoder()
        self.enc_target = self.class_la.fit_transform(target.values)
        self.base = base
        from sklearn.model_selection import train_test_split
        #元データをトレーニング用とテスト用に分割
        self.state = np.random.RandomState(1)
        self.X_train, self.X_test, self.y_train,self.y_test \
        = train_test_split(base
                         ,self.enc_target
                         ,test_size = test_rate
                         ,random_state = self.state)
    def fit(self):
        #学習実行
        self.clf.fit(self.X_train,self.y_train)
        return self.clf
    
    def fit_all(self):
        #学習実行
        self.clf.fit(self.base, self.enc_target)
        return self.clf
        
    def show_score(self):
        #分類器で予測実行
        y_pred = self.clf.predict(self.X_test)
        #テスト正解率表示
        from sklearn.metrics import accuracy_score
        print("%s 正答率 %.6f\r\n" % (self.name.encode('utf-8')
                            , accuracy_score(self.y_test,y_pred)))
    
    def test_otherdata(self,y_unknown,X_unknown):
        #学習用データとは別のラベル付きデータで正答率を確認する
        #予測
        y_pred = self.clf.predict(X_unknown)
        #テストデータのラベルを学習データと同様にエンコード
        enc_y = self.class_la.transform(y_unknown)

        #テスト正解率表示
        from sklearn.metrics import accuracy_score
        print("%s 純テストデータ正答率 %.6f\r\n" % (self.name.encode('utf-8')
                            , accuracy_score(enc_y, y_pred)))

    def predict_unknowndata(self, X_unknown):
        #未知なデータの予測
        y_pred = self.clf.predict(X_unknown)
        return y_pred
    
    def closs_vld(self, k=3, sampling_type='stkfold'):
        #交差検証
        from sklearn.model_selection import cross_val_score
        
        if sampling_type == 'shuffle':
            from sklearn.model_selection import ShuffleSplit
            sh = ShuffleSplit(train_size=k,)
            spl = sh.get_n_splits(self.base, self.enc_target)
        else:
            from sklearn.model_selection import StratifiedKFold
            stk = StratifiedKFold(n_splits =k, shuffle=True)
            spl = stk.get_n_splits(self.base, self.enc_target)
            
        cvs = cross_val_score(self.clf,self.base, self.enc_target,
                                             cv=spl, n_jobs=1)
        print(self.name + u"交差検証 k=%d" % k)
        print(cvs)
        print("avg（std）: 　%0.3f (+/- %0.3f)"
                % (cvs.mean(), cvs.std()))

    def grid_search(self, tuned_params,cv_p=5):
        #グリッドサーチによるパラメータ最適化
        print(self.name.encode('utf-8') + ' GridSearch')
        score = 'f1'
        from sklearn.grid_search import GridSearchCV
        gs_clf = GridSearchCV(
            self.clf, # 識別器
            tuned_params, # 最適化したいパラメータセット 
            cv=cv_p, # 交差検定の回数
            scoring='%s_weighted' % score ) # モデルの評価関数の指定
        gs_clf.fit(self.X_train, self.y_train)
        print(gs_clf.grid_scores_)
        print(u"最適パラメータ")
        print(gs_clf.best_params_)
        print(u'評価指標')
        from sklearn.metrics import classification_report
        y_pred = gs_clf.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))

    def grid_search_otherdata(self, tuned_params, y_unknown, X_unknown, cv_p=5):
        #初期化時とは別のラベル付きデータでグリッドサーチの結果を確認する
        print(self.name.encode('utf-8') + ' GridSearch_otherdata')
        score = 'f1'
        from sklearn.grid_search import GridSearchCV
        gs_clf = GridSearchCV(
            self.clf, # 識別器
            tuned_params, # 最適化したいパラメータセット 
            cv=cv_p, # 交差検定の回数
            scoring='%s_weighted' % score ) # モデルの評価関数の指定
        gs_clf.fit(self.base, self.enc_target)
        print(gs_clf.grid_scores_)
        print(u"最適パラメータ")
        print(gs_clf.best_params_)
        print(u'評価指標')
        from sklearn.metrics import classification_report
        y_pred = gs_clf.predict(X_unknown)
        print(classification_report(y_unknown, self.class_la.inverse_transform(y_pred)))
    
    def my_grid_search(self, tuned_params, y_unknown, X_unknown):
        #テスト用データの予測で最適化する
        #クラスにリテラルで持たせた文字列からモジュールをインポート
        exec(self.clf_str)
        #インポート文字列からクラス名を得る
        d = self.clf_str.split(" ")
        my_clf = eval("%s()" % d[len(d)-1])
        params = my_clf.get_params()
        #文字列として引数をまとめる
        paramlist = {}
        for name, vals in tuned_params.iteritems():
            p = []
            if name in params.iterkeys():
                for val in vals:
                    if type(val) is str:
                        p.append("%s='%s'," % (name,val))
                    else:
                        p.append("%s=%s," % (name,val))
                paramlist[name] = p
        
        import itertools
        para_comb = ""
        #直積で組み合わせをとる
        for pval in paramlist.itervalues():
            para_comb = pval if para_comb == "" else itertools.product(para_comb,pval)
            para_comb = ("".join(pc) for pc in para_comb)
        
        #文字列化したパラメータの組み合わせごとに学習・予測
        scores = {}
        preds = {}
        for sp in para_comb:
            print("%s,%s" % (self.name,sp))
            c= eval("%s(%s)" % (d[len(d)-1],sp))
            c.fit(self.X_train, self.y_train)
            y_pred = c.predict(X_unknown)
            y_test = self.class_la.transform(y_unknown)
            #テスト正解率表示
            from sklearn.metrics import accuracy_score
            score = accuracy_score(y_test,y_pred)
            print("%.6f\r\n" % score)
            scores[sp] = score
            preds[sp] = (y_test,y_pred)
        #ベストスコア
        for k in scores.iterkeys():
            if scores[k] == max(scores.values()):
                print("best:%.6f, %s" % (scores[k], k))
                t,p  = preds[k] #メトリクス算出用
                #メトリクス
                from sklearn.metrics import classification_report
                labels = self.class_la.inverse_transform(np.unique(self.y_train))
                print(classification_report(t, p, target_names=labels))
        
class svm_linear(single_model_base):
    """SVM線形分類"""
    def __init__(self,base,target,test_rate, C=1.0):
        from sklearn.svm import SVC
        self.clf_str = 'from sklearn.svm import SVC'
        single_model_base.__init__(self,base,target,test_rate)
        self.clf = SVC(kernel='linear', C=C, random_state=self.state
        ,probability=True)

class svm_rbf(single_model_base):
    """SVMで高次元空間への射影による分類
    RBFカーネル化を使ってみる"""
    def __init__(self,base,target,test_rate, C=10, gamma=0.001):
        from sklearn.svm import SVC
        self.clf_str = 'from sklearn.svm import SVC'
        single_model_base.__init__(self,base,target,test_rate)
        self.clf = SVC(kernel='rbf', C=C, gamma=gamma, random_state=self.state
        ,probability=True)

class svm_poly(single_model_base):
    '''SVM　多項式'''
    def __init__(self,base,target,test_rate, C=1.0, degree=2, gamma=0.001):
        from sklearn.svm import SVC
        self.clf_str = 'from sklearn.svm import SVC'
        single_model_base.__init__(self,base,target,test_rate)
        self.init_clf = SVC
        self.clf = SVC(kernel='poly', C=C, degree=degree, gamma=gamma, random_state=self.state
        ,probability=True)

class kNeighbors(single_model_base):
    """k近傍法による分類"""
    def __init__(self,base,target,test_rate, n=5, weights='uniform'):
        from sklearn.neighbors import KNeighborsClassifier
        self.clf_str = 'from sklearn.neighbors import KNeighborsClassifier'
        single_model_base.__init__(self,base,target,test_rate)    
        self.clf = KNeighborsClassifier(n, weights=weights)
        
class logistic_regression(single_model_base):
    """ロジスティック回帰による分類"""
    def __init__(self,base,target,test_rate, C=1000):
        from sklearn.linear_model import LogisticRegression
        self.clf_str = 'from sklearn.linear_model import LogisticRegression'
        single_model_base.__init__(self,base,target,test_rate)
        self.clf = LogisticRegression(C=C, random_state=self.state)
    def show_coefficients(self):
#        coef = pd.DataFrame({"Name":self.base.columns,
#                             "Coefficients":np.abs(self.clf.coef_[0])}) \
#                             .sort_values(by='Coefficients') 
        coef = pd.DataFrame(np.abs(self.clf.coef_)).T
        coef.columns = self.class_la.inverse_transform(np.unique(self.enc_target))
        name = pd.DataFrame(self.base.columns)
        coef['name'] = name        
        print(coef[:])
        coef.to_csv("D:\Python\ML\output\logistic_regr_coef.csv", index=False, encoding='shift-jis')

class decision_tree(single_model_base):
    """決定木分析による分類"""
    def __init__(self,base,target,test_rate,max_depth=3):
        from sklearn.tree import DecisionTreeClassifier
        single_model_base.__init__(self,base,target,test_rate)
        self.clf_str = 'from sklearn.tree import DecisionTreeClassifier'
        self.clf = DecisionTreeClassifier(max_depth=max_depth, random_state=self.state)

class k_means():
    """K-means法による教師なし分類"""
    def __init__(self,n_clusters=8):
        from sklearn.cluster import KMeans
        self.clf_str = 'from sklearn.cluster import KMeans'
        self.clf = KMeans(n_clusters=n_clusters)
        self.labels = []
    def clustering(self,data):
        self.clf.fit(data)
        self.labels = self.clf.labels_
        return self.labels

class AgglomerativeClustering():
    """AgglomerativeClusteringによる教師なし分類"""
    def __init__(self,n_clusters=8):
        from sklearn.cluster import AgglomerativeClustering
        self.clf_str = 'from sklearn.cluster import MeanShift'
        self.clf = AgglomerativeClustering(n_clusters=n_clusters)
        self.labels = []
    def clustering(self,data):
        self.clf.fit(data)
        self.labels = self.clf.labels_
        return self.labels   
        
class RandomForest(single_model_base):
    """ランダムフォレストによる分類"""
    def __init__(self,base,target,test_rate):
        from sklearn.ensemble import RandomForestClassifier
        single_model_base.__init__(self,base,target,test_rate)
        self.clf_str = 'from sklearn.ensemble import RandomForestClassifier'
        self.clf = RandomForestClassifier(random_state=self.state)

class GBDT(single_model_base):
    """GradientBoostingDecisionTreeによる分類"""
    def __init__(self,base,target,test_rate,n_estimators=100, learning_rate=0.1,
        max_depth=1):
        from sklearn.ensemble import GradientBoostingClassifier
        single_model_base.__init__(self,base,target,test_rate)
        self.clf_str = 'from sklearn.ensemble import GradientBoostingClassifier'
        self.clf = GradientBoostingClassifier(random_state=self.state)

class Adaboost(single_model_base):
    def __init__(self,base,target,test_rate):
        from sklearn.ensemble import AdaBoostClassifier
        single_model_base.__init__(self,base,target,test_rate)
        self.clf_str = 'from sklearn.ensemble import AdaBoostClassifier'
        self.clf = AdaBoostClassifier()

class neural_net(single_model_base):
    """ニューラルネットによる分類"""
    def __init__(self,base,target,test_rate):
        from sklearn.neural_network import MLPClassifier
        single_model_base.__init__(self,base,target,test_rate)  
        self.clf_str = 'from sklearn.neural_network import MLPClassifier'
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=self.state)
    
class Bagging(single_model_base):
    """ベース推定器を元にバギングした推定器で分類する"""
    def __init__(self,base,target,test_rate,estimater=False):
        from sklearn.ensemble import BaggingClassifier
        single_model_base.__init__(self,base,target,test_rate)
        self.clf = BaggingClassifier(estimater)   
        
class naive_bayes(single_model_base):
    """ナイーブベイズ（確率分布別）による分類"""
    def __init__(self,base,target,test_rate):
        single_model_base.__init__(self,base,target,test_rate)    
    def fit(self):
        #ナイーブベイズでの学習実行
        #正規分布
        from sklearn.naive_bayes import GaussianNB 
        nb_g = GaussianNB()
        nb_g.fit(self.X_train,self.y_train)
        #{名前:分類器}の辞書型にして返す
        nb = {"gaussian" : nb_g}
        #ベルヌーイ分布
        from sklearn.naive_bayes import BernoulliNB
        nb_b = BernoulliNB()
        nb_b.fit(self.X_train,self.y_train)
        nb["bernoulli"] = nb_b
        #多項分布
        from sklearn.naive_bayes import MultinomialNB
        nb_m = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
        nb_m.fit(self.X_train,self.y_train)
        nb["multinomial"] = nb_m
        return nb
