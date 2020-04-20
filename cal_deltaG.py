#!/bin/python
import os

def cal_dG(fasta_file):
    out=os.popen("echo -e \"%s\nasint1x2.dat\nasint2x3.dat\ndangle.dat\nloop.dat\nmiscloop.dat\nsint2.dat\nsint4.dat\nsint6.dat\nstack.dat\ntloop.dat\ntriloop.dat\ntstackh.dat\ntstacki.dat\"| quikfold | less" % fasta_file)
#print(out.read().split("\n"))
    n=0
    deltaG_lst=[]
    for line in out.readlines():
        line=line.strip()
    #print(line) 
        if line!=" " and line!="":
            jud=line[1]
            if jud==".":
                deltaG=line.split("=")[1].strip().split(" ")[0]
                deltaG_lst.append(float(deltaG))
    return deltaG_lst

#seq.fasta
