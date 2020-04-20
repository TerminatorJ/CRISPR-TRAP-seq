#!/bin/python

import os
import sys
import pandas as pd
import numpy as np
import time
import sklearn
import numpy as np
import Bio.SeqUtils as SeqUtil
import Bio.Seq as Seq
# import azimuth.util
import sys
import Bio.SeqUtils.MeltingTemp as Tm
import pickle
import itertools
from sklearn.preprocessing import MinMaxScaler
import sklearn
import sklearn.linear_model
from sklearn.grid_search import GridSearchCV
import sklearn.ensemble as en
import scipy.stats as st
import scipy as sp
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor








def spearmanr(x,y):
    #x=x.flatten()
    #y=y.flatten()
    r, p = st.spearmanr(x, y)
    return r, p

def spearman_scoring(clf,X, y):
    y_pred = clf.predict(X).flatten()
    return sp.stats.spearmanr(y_pred, y)[0]
def extract_spearman_for_fold_deep(metrics,test, y_pred):
    spearman=spearmanr(test, y_pred)[0]
#     assert not np.isnan(spearman), "found nan spearman"
    metrics.append(spearman)

def select_model(x_deep,y_deep,model,state,target_num):
    if model== "Ada":
        print("Adaboost with KFold")
        param_grid = {'learning_rate': [0.1, 0.05, 0.01]}
        n_folds=10
        metrics=[]
        kf = KFold(n_splits=10)

        n=0
        for train,test in kf.split(x_deep):
            n+=1
            x_train=x_deep[train]
            y_train=y_deep[train]
            x_test=x_deep[test]
            y_test=y_deep[test]
            
            cv = sklearn.cross_validation.KFold(x_train.shape[0], n_folds=n_folds, shuffle=True)
            est = en.GradientBoostingRegressor()
            clf_1 = GridSearchCV(est, param_grid, n_jobs=-1, verbose=1, cv=cv, scoring=spearman_scoring, iid=False)
            clf_1.fit(x_train,y_train)
            #pickle.dump(clf_1.best_params_,open(para_path+"%d_Ada_%d_best_params.pickle" % (target_num,n),"wb"))
#             print("Ada best params:"+str(clf_1.best_params_))
            #with open(model_path+str(target_num)+"_"+str(n)+"_Ada.pickle","wb") as f2:
            #    pickle.dump(clf_1,f2)
            y_pred = clf_1.predict(x_test)
            extract_spearman_for_fold_deep(metrics,y_test,y_pred)
        root=os.getcwd()
        pj=lambda *path: os.path.abspath(os.path.join(*path)) 
        pickle.dump(metrics,open(pj(root,"result/Ada_%d_metrics.pickle" % target_num),"wb"))
        return metrics
    elif model=="DT":
        print("DT With KFold")
        parameters = {"max_depth":(1,2,5,10,100,1000)}
        n_folds=10
        metrics=[]       
        kf = KFold(n_splits=10)
        n=0
        for train,test in kf.split(x_deep):
            n+=1
            x_train=x_deep[train]
            y_train=y_deep[train]
            x_test=x_deep[test]
            y_test=y_deep[test]
            cv = sklearn.cross_validation.KFold(x_train.shape[0], n_folds=n_folds, shuffle=True)
            DT = tree.DecisionTreeRegressor(random_state=state)
            clf_2 = GridSearchCV(DT, parameters, n_jobs=-1, verbose=1, cv=cv, scoring=spearman_scoring, iid=False)
            clf_2.fit(x_train,y_train)
            #pickle.dump(clf_2.best_params_,open(para_path+"%d_DT_%d_best_params.pickle" % (target_num,n),"wb"))
#             print("DT best params:" % +str(clf_2.best_params_))
            #with open(model_path+str(target_num)+"_"+str(n)+"_DT.pickle","wb") as f2:
            #    pickle.dump(clf_2,f2)
            #    print("after fitting")
            y_pred = clf_2.predict(x_test)
            extract_spearman_for_fold_deep(metrics,y_test,y_pred)
        root=os.getcwd()
        pj=lambda *path: os.path.abspath(os.path.join(*path))
        pickle.dump(metrics,open(pj(root,"result/DT_%d_metrics.pickle" % target_num),"wb"))
        return metrics
    elif model=="linear_model":
        print("linear_model With KFold")
        n_folds=10
        metrics=[]
        reg = linear_model.LinearRegression()
        n=0
        kf = KFold(n_splits=10)
        for train,test in kf.split(x_deep):
            n+=1
            x_train=x_deep[train]
            y_train=y_deep[train]
            x_test=x_deep[test]
            y_test=y_deep[test]
            reg.fit(x_train,y_train)
            #with open(model_path+str(target_num)+"_"+str(n)+"_linear_reg.pickle","wb") as f2:
            #    pickle.dump(reg,f2)
            #    print("after fitting")
            y_pred = reg.predict(x_test)
            extract_spearman_for_fold_deep(metrics,y_test,y_pred)
        root=os.getcwd()
        pj=lambda *path: os.path.abspath(os.path.join(*path))
        pickle.dump(metrics,open(pj(root,"result/LR_%d_metrics.pickle" % target_num),"wb"))
        return metrics
    elif model=="Ridge":
        print("Ridge With KFold")
        parameters = {"alpha":(1,3,5,7,9,10)}
        n_folds=10
        metrics=[]
        kf = KFold(n_splits=10)
        n=0
        for train,test in kf.split(x_deep):
            n+=1
            x_train=x_deep[train]
            y_train=y_deep[train]
            x_test=x_deep[test]
            y_test=y_deep[test]
            cv = sklearn.cross_validation.KFold(x_train.shape[0], n_folds=n_folds, shuffle=True)
            reg = linear_model.Ridge()
            clf_3 = GridSearchCV(reg, parameters, n_jobs=-1, verbose=1, cv=cv, scoring=spearman_scoring, iid=False)
            clf_3.fit(x_train,y_train)
            #pickle.dump(clf_3.best_params_,open(para_path+"%d_Ridge_%d_best_params.pickle" % (target_num,n),"wb"))
#             print("%s best params:" % (save_name_tail)+str(clf_3.best_params_))
            #with open(model_path+str(target_num)+"_"+str(n)+"_Ridge_reg.pickle","wb") as f2:
            #    pickle.dump(clf_3,f2)
            #    print("after fitting")
            y_pred = clf_3.predict(x_test)
            extract_spearman_for_fold_deep(metrics,y_test,y_pred)
        root=os.getcwd()
        pj=lambda *path: os.path.abspath(os.path.join(*path))
        pickle.dump(metrics,open(pj(root,"result/Ridge_%d_metrics.pickle" % target_num),"wb"))
        return metrics
    elif model=="Lasso":
        print("Lasso With KFold")
        parameters = {"alpha":(0.1,0.5,0.75,1,10)}
        n_folds=10
        metrics=[]
        print("doing the GridSearchCV")
        kf = KFold(n_splits=10)
        n=0
        for train,test in kf.split(x_deep):
            n+=1
            x_train=x_deep[train]
            y_train=y_deep[train]
            x_test=x_deep[test]
            y_test=y_deep[test]
            cv = sklearn.cross_validation.KFold(x_train.shape[0], n_folds=n_folds, shuffle=True)
            reg = linear_model.Lasso()
            clf_4 = GridSearchCV(reg, parameters, n_jobs=-1, verbose=1, cv=cv, scoring=spearman_scoring, iid=False)
            clf_4.fit(x_train,y_train)
            #pickle.dump(clf_4.best_params_,open(para_path+"%d_Lasso_%d_best_params.pickle" % (target_num,n),"wb"))
#             print("%s best params:" % (save_name_tail)+str(clf_4.best_params_))
            #with open(model_path+str(target_num)+"_"+str(n)+"_Lasso_reg.pickle","wb") as f2:
            #    pickle.dump(clf_4,f2)
            #    print("after fitting")
            y_pred = clf_4.predict(x_test)
            extract_spearman_for_fold_deep(metrics,y_test,y_pred)
        root=os.getcwd()
        pj=lambda *path: os.path.abspath(os.path.join(*path))
        pickle.dump(metrics,open(pj(root,"result/Lasso_%d_metrics.pickle" % target_num),"wb"))
        return metrics
    elif model=="Bayes Ridge":
        print("Bayes Ridge With KFold")
        metrics=[]
        reg = linear_model.BayesianRidge()
        print("doing the GridSearchCV")
        n=0
        kf = KFold(n_splits=10)
        for train,test in kf.split(x_deep):
            n+=1
            x_train=x_deep.iloc[train]
            y_train=y_deep.iloc[train]
            x_test=x_deep.iloc[test]
            y_test=y_deep.iloc[test]
            reg.fit(x_train,y_train)
            #with open(model_path+str(trap_typ)+"_fold"+str(n)+"_BRR.pickle","wb") as f2:
            #    pickle.dump(reg,f2)
            #    print("after fitting")
            y_pred = reg.predict(x_test)
            extract_spearman_for_fold_deep(metrics,y_test,y_pred)
        root=os.getcwd()
        pj=lambda *path: os.path.abspath(os.path.join(*path))
        pickle.dump(metrics,open(pj(root,"result/BRR_%s_metrics.pickle" % trap_typ),"wb"))
        return metrics
    elif model== "RF":
        print("RF With KFold")
        metrics=[]
        rf =RandomForestRegressor()
        print("doing the GridSearchCV")
        kf = KFold(n_splits=10)
        n=0
        for train,test in kf.split(x_deep):
            n+=1
            x_train=x_deep[train]
            y_train=y_deep[train]
            x_test=x_deep[test]
            y_test=y_deep[test]
            rf.fit(x_train,y_train)
            #with open(model_path+str(target_num)+"_"+str(n)+"_RF.pickle","wb") as f2:
            #    pickle.dump(rf,f2)
            #    print("after fitting")
            y_pred = rf.predict(x_test)
            extract_spearman_for_fold_deep(metrics,y_test,y_pred)
        root=os.getcwd()
        pj=lambda *path: os.path.abspath(os.path.join(*path))
        pickle.dump(metrics,open(pj(root,"result/RF_%d_metrics.pickle" % target_num),"wb"))
        return metrics
    elif model == "NN":
        print("NN With KFold")
        metrics=[]
        nn=MLPRegressor(hidden_layer_sizes=[100,100],activation="relu",alpha=1.0,solver="lbfgs")
        kf = KFold(n_splits=10)
        n=0
        for train,test in kf.split(x_deep):
            n+=1
            x_train=x_deep[train]
            y_train=y_deep[train]
            x_test=x_deep[test]
            y_test=y_deep[test]
            nn.fit(x_train,y_train)
            #with open(model_path+str(target_num)+"_"+str(n)+"_NN.pickle","wb") as f2:
            #   pickle.dump(nn,f2)
            #    print("after fitting")
            y_pred = nn.predict(x_test)
            extract_spearman_for_fold_deep(metrics,y_test,y_pred)
        root=os.getcwd()
        pj=lambda *path: os.path.abspath(os.path.join(*path))
        pickle.dump(metrics,open(pj(root,"result/NN_%d_metrics.pickle" % target_num),"wb"))
        return metrics



def get_target(efficiency_path,typ):
    if typ=="ABE":
        this_table=pd.read_excel(efficiency_path,header=0,sheet_name="Sheet2")
        N13=this_table["N13(N3)"]
        N14=this_table["N14(N4)"]
        N15=this_table["N15(N5)"]
        N16=this_table["N16(N6)"]
        N17=this_table["N17(N7)"]
        N18=this_table["N18(N8)"]
        N19=this_table["N19(N9)"]
        N20=this_table["N20(N10)"]
        N21=this_table["N21(N11)"]
        return N3,N4,N5,N6,N7,N8,N9,N10,N11
    elif typ=="CBE":
        site_list=["N%d" % i for i in range(1,21)]
        sv_site_dict={}
        this_table=pd.read_excel(efficiency_path,header=0)
        for site in site_list:
            sv_site_dict[site]=this_table[site]
        return sv_site_dict
    elif typ=="AIO":
        need_col=["#spCas9-507","#ABE-515","#CBE-508"]
        dict_key=["Cas9","ABE","CBE"]
        this_table=pd.read_excel(efficiency_path)
        target_dict={}
        for col,ed_typ in zip(need_col,dict_key):
            eff=this_table[col]
            target_dict[ed_typ]=eff
        return target_dict
            
##main
def main(typ):
    root=os.getcwd()
    pj=lambda *path: os.path.abspath(os.path.join(*path)) 
    if typ=="ABE_site":
        efficiency_path=pj(root,"data/ABE_efficiency.xlsx")
        feature_mt_path=pj(root,"feature_mt/feature_matrix_ABE_all.pickle")
        target_dict=get_target(efficiency_path,typ="ABE")
        feature_mt=pickle.load(open(pj(root,feature_mt_path),"rb"))
        need_site_dict=pickle.load(open(pj(root,"need_site/ABE_need_site_dict.pickle"),"rb"))
        n=12
        for site,target in target_dict.items():
            n+=1
            need_site=need_site_dict[site]
            target=target.iloc[list(need_site)]
            feature_mt_1=feature_mt.iloc[list(need_site)]
            remove_feat_list=["A_3","A_4","A_5","A_6","A_7","A_8","A_9","A_10","A_11"]
            rmv_n=n-13
            remove_feat=remove_feat_list[rmv_n]
            feature_mt_1=feature_mt_1.drop(remove_feat,1)
            br_metrics=select_model(feature_mt_1,target,"Bayes Ridge",state=0,target_num=n-10)
    elif typ=="CBE_site":
        efficiency_path=pj(root,"data/CBE_efficiency.xlsx")
        feature_mt_path=pj(root,"feature_mt/feature_matrix_CBE.pickle")
        target_dict=get_target(efficiency_path,typ="CBE")
        feature_mt=pickle.load(open(pj(root,feature_mt_path),"rb"))
        need_site_dict=pickle.load(open(pj(root,"need_site/CBE_need_site_dict.pickle"),"rb"))
        n=0
        for site,target in target_dict.items():
            need_site=need_site_dict[site]
            filtered_mt_1=feature_mt.iloc[list(need_site)]
            target=target.iloc[list(need_site)]
            remove_feat_list=["C_%d" % i for i in range(1,21)]
            remove_feat=remove_feat_list[n]
            filtered_mt_1=filtered_mt_1.drop(remove_feat,1)
            br_metrics=select_model(filtered_mt_1,target,"Bayes Ridge",state=0,target_num=n+1)
            n+=1
    elif typ=="all":
        efficiency_path=pj(root,"data/ABE_CBE_Cas9_efficiency.xlsx")
        feature_mt_path=pj(root,"feature_mt/feature_matrix_ABE_all.pickle")
        target_dict=get_target(efficiency_path,typ="AIO")
        feature_mt=pickle.load(open(pj(root,feature_mt_path),"rb"))
        need_site_dict=pickle.load(open(pj(root,"need_site/alltyp_need_dict.pickle"),"rb"))
        for typ,target in target_dict.items():
            need_site=need_site_dict[typ]
            target=target.iloc[list(need_site)]
            feature_mt_1=feature_mt.iloc[list(need_site)]
            br_metrics=select_model(feature_mt_1,target,"Bayes Ridge",state=0,target_num=typ)
     














    







