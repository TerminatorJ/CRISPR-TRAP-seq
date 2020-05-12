#!/bin/python
import pickle
import os
import model
import numpy as np
import pandas as pd
import featurization
def predict(this_input,typ,full_length=True,site=None):
    if typ=="ABE" and full_length==True:
        model=pickle.load(open("./model/full-length/ABE_model.pickle","rb"))
    elif typ=="CBE" and full_length==True:
        model=pickle.load(open("./model/full-length/CBE_model.pickle","rb"))
    elif typ=="Cas9" and full_length==True:
        model=pickle.load(open("./model/full-length/Cas9_model.pickle","rb"))
    elif typ=="ABE" and full_length==False:
        model=pickle.load(open("./model/single_site/ABE_N%d.pickle" % site,"rb"))
    elif typ=="CBE" and full_length==False:
        model=pickle.load(open("./model/single_site/CBE_N%d.pickle" % site,"rb"))
    pred=model.predict(this_input)
    return pred
def val_pam(array_like):
    seq_list=[]
    for i in array_like:
        assert i[25:27]=="GG","The [25:27] bases of input sequences should be GG"
        assert len(i)==30,"The length of input sequences should be 30mer!!!"
def get_score(fasta_file,typ_,full_length=True,site=None):
    new_str=""
    all_list=[]
    seq30=[]
    with open(fasta_file,"r") as f1:
        fa_str=f1.read()
        fa_lst=fa_str.split(">")[1:]
        seq_lst=[i.split("\n")[1] for i in fa_lst]
        val_pam(seq_lst)
        for index,ele in enumerate(fa_lst):  
            num=index+1
            seq=ele.split("\n")[1]
            seq30.append(seq)
            new_ele=">"+ele
            new_str+=new_ele
            if num%9==0:##because the quikfold can just run the pipline maximum 9 jobs!!!
                
                input_matrix=featurization.get_input(new_str)
                output=predict(this_input=input_matrix,typ=typ_,full_length=full_length,site=site)
                all_list+=list(output)
                new_str=""
            elif num==len(fa_lst) or (len(fa_lst)-num)==(len(fa_lst)%9-1):
                new_str=""
                for k,i in enumerate(fa_lst[index:]): 
                    seq=i.split("\n")[1]
                    if num!=len(fa_lst)-len(fa_lst)%9+1:#the first item of interaction
                        seq30.append(seq)
                    ele=">"+i
                    num+=1
                    new_str+=ele
                input_matrix=featurization.get_input(new_str)
                output=predict(input_matrix,typ=typ_,full_length=full_length,site=site)
                all_list+=list(output)
                break
    rsl=pd.DataFrame({"30mer":seq30,"GNL-Scorer":all_list})
    rsl.to_csv("%s_GNL_result.csv" % typ_,sep=",")
    
