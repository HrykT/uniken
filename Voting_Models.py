# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 21:25:31 2016

@author: Hryk
"""

from Single_Models import single_model_base

class Voting_Model(single_model_base):
    """アンサンブル学習　Voting推定器クラス"""
    def __init__(self,base,target,test_rate,clf_list, weights=None):
        from sklearn.ensemble import VotingClassifier
        single_model_base.__init__(self,base,target,test_rate)
        self.clf = VotingClassifier(clf_list, voting='soft', weights=weights)