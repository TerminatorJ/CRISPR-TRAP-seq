
# coding: utf-8

# # Featurized the sgRNA

# In[1]:


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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from cal_deltaG import cal_dG
import scipy as sp


# In[17]:



def cal_deltaG(seq_file,cal_deltaG_dir,input_seq=False):
    if input_seq == False:
        items=seq_file.split(">")[1:]
        seqs=[]
        for item in items:
            item=item.strip()
            seq=item.split("\n")[1]
            seqs.append(seq)
        get_20mer=lambda x : x[4:24]
#     map(get_20mer,seqs)
        seqs_20mer=[]
        for i in map(get_20mer,seqs):
            seqs_20mer.append(i)
        for_deltaG=";".join(seqs_20mer)
    
    elif input_seq == True:
        #seq_file
        #seqs_20mer=[]
        #for argv in sys.argv[1:]:
        #    assert len(argv)==30, "the sequence input must be 30mer"
        #    seqs_20mer.append(argv)
        for_deltaG=";".join(seq_file)
    for_deltaG_dir=os.path.join(cal_deltaG_dir,"for_deltaG.txt")
    with open(for_deltaG_dir,"w") as f2:
        f2.write(for_deltaG)
#     for_deltaG_dir=os.path.join(current_dir,)
    deltaG_result_dir=os.path.join(cal_deltaG_dir,"deltaG_result.txt")
    os.system("hybrid-ss-min -E %s -o %s" % (for_deltaG_dir,deltaG_result_dir))
    if os.path.exists(deltaG_result_dir)=="False":
        print("the %s is not exists" % deltaG_result_dir)
    return deltaG_result_dir+".dG"    
start=time.time()
##featurazation
def get_alphabet(order,raw_alphabet=["A","G","C","T"]):##
    alphabet=["".join(i) for i in itertools.product(raw_alphabet,repeat=order)]
    return alphabet
def genera_nucleotide_features(s,order,prefix="",feature_type="all",raw_alphabet=["A","T","G","C"],NGGX=False):##
    alphabet=get_alphabet(order,raw_alphabet=raw_alphabet)
    features_pos_dependent=np.zeros(len(alphabet)*(len(s)-(order-1)))
    features_pos_independent=np.zeros(np.power(len(raw_alphabet),order))
    index_dependent=[]
    index_independent=[]
    
    for position in range(0,(len(s)-order+1),1):
        if NGGX==False:
            for i in alphabet:
#                 exg=position-3
                if position<=3:
                    exg=position-4
                elif position>3:
                    exg=position-3
                
                index_dependent.append("%s%s_%s" % (prefix,i,exg))
        else:
            for i in alphabet:
                index_dependent.append("NGGX%s%s_%s" % (prefix,i,position))
    for i in alphabet:
        index_independent.append("%s%s" % (prefix,i))
    for position in range(0,(len(s)-order+1),1):
        nucl=s[position:position+order]
        features_pos_dependent[alphabet.index(nucl)+(position*len(alphabet))]=1.0
        features_pos_independent[alphabet.index(nucl)]+=1.0
    if feature_type=="all" or feature_type=="pos_independent":
        if feature_type=="all":
            res=pd.Series(features_pos_dependent,index=index_dependent),pd.Series(features_pos_independent,index=index_independent)
            return res
        else:
            res=pd.Series(features_pos_independent,index=index_independent)
            return res
    res=pd.Series(features_pos_dependent,index=index_dependent)#
    return res
def nucleotide_features_dictionary(prefix=""):
    seqname=["-4","-3","-2","-1"]
    seqname.extend([str(i) for i in range(1,21)])#
    seqname.extend(["N","G","G","+1","+2","+3"])
    orders=[1,2,3]
    sequence_len=30
    feature_names_dep=[]
    feature_names_indep=[]
    index_dependent=[]
    index_independent=[]
    for  order in orders:
        raw_alphabet=["A","G","C","T"]
        alphabet=["".join(i)for i in itertools.product(raw_alphabet,repeat=order)]
        feature_pos_dependent=np.zeros(len(alphabet)*(sequence_len-order+1))
        feature_pos_independent=np.zeros(np.power(len(raw_alphabet),order))
        index_dependent.extend(["%s_pd.Order%s_P%d" % (prefix,order,i) for i in range(len(feature_pos_dependent))])
        index_independent.extend(["%s_pi.Order%s_P%d" % (prefix,order,i) for i in range(len(feature_pos_independent))])
        for pos in range(sequence_len-(order-1)) :
            for letter in alphabet:
                feature_names_dep.append("%s_%s" % (letter,seqname[pos]))
        for letter in alphabet:
            feature_names_indep.append("%s" % letter)
    index_all=index_dependent+index_independent
    feature_all=feature_names_dep+feature_names_indep
    return dict(zip(index_all,feature_all))#

def NGGX_interaction_feature(s):
    sequence=s
    NX=sequence[25]+sequence[27]
    NX_onehot=genera_nucleotide_features(NX,order=2,feature_type="pos_dependent",NGGX=True)
#     feat_NX=pandas.concat([feat_NX,NX_onehot],axis=1)
    return NX_onehot
def apply_NGGX_feature(data):
    NGGX_feat=data["30mer"].apply(NGGX_interaction_feature)
    return NGGX_feat
def gc_percent(seq):
    return (seq.count('G') + seq.count('C'))/float(len(seq))
def countGC_20mer(s, length_audit=True):
    if length_audit:
        assert len(s) == 30, "seems to assume 30mer"
    return s.count("G")+s.count("C")
def gc_features(data, audit=True):
    gc_count=data["30mer"].apply(lambda seq:countGC_20mer(seq, audit))
    gc_count.name="gc_count"
    gc_above_10=(gc_count>10)*1
    gc_above_10.name="gc_above_10"
    gc_below_10=(gc_count<10)*1
    gc_below_10.name="gc_below_10"
    data["gc_count"]=gc_count
    data["gc_above_10"]= gc_above_10 
    data["gc_below_10"]=gc_below_10
    return gc_above_10, gc_below_10, gc_count
def filter_no_30mer(data):
    data["30mer"]=data["30mer"].apply(lambda x: x[0:30])
    return data
def Tm_feature(data,pam_audit=True,learn_options=None):
    if learn_options is None or "Tm segments" not in learn_options:
        segments=[(19,24),(11,19),(6,11),(4,24)]
    else:
        segments=learn_options["Tm segments"]
    sequence=data["30mer"].values
    featarray=np.ones((sequence.shape[0],5))
    rna=True
    for i,seq in  enumerate(sequence):
        if pam_audit and seq[25:27]!="GG":
            continue
            raise Exception("excepted GG but found %s" % seq[25:27])
        
        featarray[i,0]=Tm.Tm_staluc(seq,rna=rna)#30mer
        featarray[i,1]=Tm.Tm_staluc(seq[segments[0][0]:segments[0][1]], rna=rna) #5nts immediately proximal of the NGG PAM
        featarray[i,2]=Tm.Tm_staluc(seq[segments[1][0]:segments[1][1]], rna=rna)   #8-mer
        featarray[i,3]=Tm.Tm_staluc(seq[segments[2][0]:segments[2][1]], rna=rna)      #5-mer
        featarray[i,4]=Tm.Tm_staluc(seq[segments[3][0]:segments[3][1]], rna=rna) #20-spacer
    feat=pd.DataFrame(featarray,index=data.index,columns=["Tm global_30mer%s" % rna, "5mer_end_%s" % rna, "8mer_middle_%s" % rna, "5mer_start_%s" % rna,"Tm global_spacer_%s" % rna])                            
    return feat





def apply_nucleotide_features(seq_data_frame,order,include_pos_independent,prefix=""):                                                                                                              
    if include_pos_independent:
        feat_pd = seq_data_frame.apply(genera_nucleotide_features, args=(order, prefix, 'pos_dependent'))
        feat_pi = seq_data_frame.apply(genera_nucleotide_features, args=(order, prefix, 'pos_independent'))
        return feat_pd,feat_pi 
    else:
        feat_pd = seq_data_frame.apply(genera_nucleotide_features, args=(order, prefix, 'pos_dependent'))
        return feat_pd
def get_all_order_nuc_features(data,feature_sets,maxorder, max_index_to_use, prefix=""): 
    for order in range(1, maxorder+1):
        nuc_features_pd, nuc_features_pi = apply_nucleotide_features(data, order,prefix=prefix)
        return  nuc_features_pd, nuc_features_pi
def concat_all_need_feat(data):
    
    gc_count=gc_features(filter_no_30mer(data), audit=True)[2]##Series
    gc_above_10=gc_features(filter_no_30mer(data), audit=True)[0]##Series
    gc_below_10=gc_features(filter_no_30mer(data), audit=True)[1]##Series
    pair_base_pd=apply_nucleotide_features(filter_no_30mer(data)["30mer"],order=2,prefix="",include_pos_independent=False)##df
    single_base_pd=apply_nucleotide_features(filter_no_30mer(data)["30mer"],order=1,prefix="",include_pos_independent=False)##df
    pair_base_pi=apply_nucleotide_features(filter_no_30mer(data)["30mer"],order=2,prefix="",include_pos_independent=True)[1]##tuple
#     return pair_base_pi
    single_base_pi=apply_nucleotide_features(filter_no_30mer(data)["30mer"],order=1,prefix="",include_pos_independent=True)[1]##tuple
#     return single_base_pi
#     return type(single_base_pi)
    Tm=Tm_feature(filter_no_30mer(data),pam_audit=True,learn_options=None)##df
    NGGX=apply_NGGX_feature(data)
#     return type(Tm)
    all_need_feat=pd.concat([gc_count,gc_above_10,gc_below_10,pair_base_pd,single_base_pd,pair_base_pi,single_base_pi,Tm,NGGX],axis=1)
    return all_need_feat
##add other features
def five_continuous_base(data):
    sequence=data["30mer"].values
    five_con_A=np.zeros(len(sequence))
    five_con_G=np.zeros(len(sequence))
    five_con_C=np.zeros(len(sequence))
    five_con_T=np.zeros(len(sequence))
    for index,seq in enumerate(sequence):
        if "AAAAA" in seq.upper():
            five_con_A[index]=1
        elif "TTTTT" in seq.upper():
            five_con_T[index]=1
        elif "GGGGG" in seq.upper():
            five_con_G[index]=1
        elif "CCCCC" in seq.upper():
            five_con_C[index]=1
    all_continue=np.vstack((five_con_A,five_con_G,five_con_C,five_con_T))
    
    add_five_con=pd.DataFrame(all_continue.transpose(),index=data.index,columns=["AAAAA", "TTTTT","GGGGG","CCCCC"])
    return add_five_con
#     add_five_con=pd.concat([data,continue_feat],axis=1)

#     return add_five_con
def apply_three_contiu(seq_data_frame,order,include_pos_independent=True,prefix=""):                                                                                                              
    if include_pos_independent:
        feat_pd_three_contiu = seq_data_frame.apply(genera_nucleotide_features, args=(order, prefix, 'pos_dependent'))
#         return feat_pd
        feat_pi_three_contiu = seq_data_frame.apply(genera_nucleotide_features, args=(order, prefix, 'pos_independent'))
        
        return feat_pd_three_contiu,feat_pi_three_contiu
#     else:
#         feat_pd = seq_data_frame.apply(genera_nucleotide_features, args=(order, prefix, 'pos_dependent'))
#         assert not np.any(np.isnan(feat_pd)), "found nan in feat_pd"
        return feat_pd
def deltaG_20mer(data,fasta_name):#file_name shoud be the type of fasta.
    deltaG_lst=cal_dG(fasta_name)
    deltaG_df=pd.DataFrame(deltaG_lst,index=data.index)
    deltaG_df.columns=["DeltaG"]
    return deltaG_df
def concat_all_need_feat_second(data,fasta_name):
    feat_pd_three_contiu=apply_three_contiu(filter_no_30mer(data)["30mer"],order=3,include_pos_independent=True,prefix="")[0]
    feat_pi_three_contiu=apply_three_contiu(filter_no_30mer(data)["30mer"],order=3,include_pos_independent=True,prefix="")[1]
    deltaG_df=deltaG_20mer(data,fasta_name)
    all_need_feat_second=pd.concat([concat_all_need_feat(data),feat_pi_three_contiu,feat_pd_three_contiu,deltaG_df],axis=1)
    return all_need_feat_second
def series_to_df(Series):
    df=pd.DataFrame(data=Series,columns=["30mer"])
    return df


def fasta_2_df(fasta_file):
    items=fasta_file.split(">")[1:]
    seqs=[]
    for item in items:
        item=item.strip()
        seq=item.split("\n")[1]
        seqs.append(seq)
    seq_df=pd.DataFrame(seqs,columns=["30mer"]) 
    return seq_df
def list_2_df(list_type):
    seq_df=pd.DataFrame(list_type,columns=["30mer"])
    return seq_df

def list_2_fasta(list_type):
    all_str=""
    for index,seq in enumerate(list_type):
        line_str_1=">"+str(index+1)+"\n"
        line_str_2=seq+"\n"
        item=line_str_1+line_str_2
        all_str+=item
    with open("./seq_fasta","w") as f1:
        f1.write(all_str)
        


def get_deltaG(fasta):
    deltaG_lst=cal_deltaG(fasta)    
    return deltaG_list
def save_20mer_fasta(fasta_name_30mer):
    with open(fasta_name_30mer,"r") as f2:#this input is a name of fasta file contain sequences
        data=f2.read()
    items=data.split(">")[1:]
    seqs=[]
    new_str=""
    for item in items:
        item=item.strip()
        seq=item.split("\n")[1]
        head=item.split("\n")[0]
        need_20mer=seq[4:24]
        new_str+=">"+head+"\n"+need_20mer+"\n"
        
    with open("seq_20mer.fasta","w") as f3:
        f3.write(new_str)
    

def get_input(fasta_name_30mer):
    with open(fasta_name_30mer,"r") as f1:#this input is a name of fasta file contain sequences
        data=f1.read()
    data_df=fasta_2_df(data)
    save_20mer_fasta(fasta_name_30mer)
    this_input=concat_all_need_feat_second(data_df,"seq_20mer.fasta")#input_ should be aranged in the file of cal_deltaG
    return this_input


