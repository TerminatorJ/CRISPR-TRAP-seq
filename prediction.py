#!/bin/python
import pickle
import os
import model
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
    
