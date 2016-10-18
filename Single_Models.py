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
        from sklearn.cross_validation import train_test_split
        #元データをトレーニング用とテスト用に分割
        self.X_train, self.X_test, self.y_train,self.y_test \
        = train_test_split(base
                         ,self.enc_target
                         ,test_size = test_rate
                         ,random_state = 0)
    def fit(self):
        #学習実行
        self.clf.fit(self.X_train,self.y_train)
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
        
    def closs_vld(self):
        #交差検証
        from sklearn import cross_validation
        kfl = cross_validation.KFold(n=len(self.enc_target), n_folds=3,
                                     shuffle=True)
        cvs = cross_validation.cross_val_score(self.clf,
                                         self.base, self.enc_target,
                                         cv=kfl, n_jobs=1)
        print(self.name + u"交差検証 k=%d" % 3)
        print(cvs)
        print("avg（std）: 　%0.3f (+/- %0.3f)"
                % (cvs.mean(), cvs.std()))

    def grid_search(self, tuned_params,cv_p=5):
        #グリッドサーチによるパラメータ最適化
        print(self.name.encode('utf-8') + 'GridSearch')
        score = 'f1'
        from sklearn.grid_search import GridSearchCV
        gs_clf = GridSearchCV(
            self.clf, # 識別器
            tuned_params, # 最適化したいパラメータセット 
            cv=cv_p, # 交差検定の回数
            scoring='%s_weighted' % score ) # モデルの評価関数の指定
        gs_clf.fit(self.X_train, self.y_train)
        print(gs_clf.grid_scores_)
        print(u'評価指標')
        from sklearn.metrics import classification_report
        y_pred = gs_clf.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))

class svm_linear(single_model_base):
    """SVM線形分類"""
    def __init__(self,base,target,test_rate):
        from sklearn.svm import SVC
        single_model_base.__init__(self,base,target,test_rate)
        self.clf = SVC(kernel='linear', C=1.0,random_state=44)

class svm_rbf(single_model_base):
    """SVMで高次元空間への射影による分類
    RBFカーネル化を使ってみる"""
    def __init__(self,base,target,test_rate):
        from sklearn.svm import SVC
        single_model_base.__init__(self,base,target,test_rate)
        self.clf = SVC(kernel='rbf', C=1.0, gamma=0.2, random_state=5)

class svm_poly(single_model_base):
    '''SVM　多項式'''
    def __init__(self,base,target,test_rate):
        from sklearn.svm import SVC
        single_model_base.__init__(self,base,target,test_rate)
        self.clf = SVC(kernel='poly', C=1.0, random_state=5)

class kNeighbors(single_model_base):
    """k近傍法による分類"""
    def __init__(self,base,target,test_rate):
        from sklearn import neighbors
        single_model_base.__init__(self,base,target,test_rate)    
        self.clf = neighbors.KNeighborsClassifier(5, weights='uniform')
        
class logistic_regression(single_model_base):
    """ロジスティック回帰による分類"""
    def __init__(self,base,target,test_rate):
        from sklearn.linear_model import LogisticRegression
        single_model_base.__init__(self,base,target,test_rate)    
        self.clf = LogisticRegression()
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
