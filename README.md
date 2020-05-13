# CRISPR TRAP-seq
This is a ML learning model to predict the gRNA efficiency  of trap-based editing system. eg.ABE, CBE, Cas9

# Environment: Linux
# Packages requirement: 
python>=3.5,biopython,pandas,sklearn,os,sys,numpy,pickle,itertools,scipy,matplotlib
# DeltaG calculation (environmental preparation)

To get the binary file of mfold, please download the package from the website: http://unafold.rna.albany.edu/?q=mfold/download-mfold

```bash
#untar the tar.gz file
tar -zxvf mfold-3.6.tar.gz
```
# install mfold

```bash
./configure prefix="$dir_you_want_to_see_the_bin"
make
make install
#then you can see the bin file in the dir you set in the $prefix
```
Exporting the "quikfold" into your PATH (note:quikfold can help you to calculate the deltaG of multiple sequences at the same time!!! )

## Adding the following code into your ~/.bashrc

```bash
export PATH=$your_bin_dir_of_mfold:$PATH
```

Before calculating the deltaG, you should prepare a fasta file, which contain 30mer(4 bp+23 bp+3 bp)sequence
note: you should put this file into the working direction under $CRISPR TRAP-seq

your direction order should like this:

```bash
CRISPR TRAP-seq
...asint1x2.dat
...loop.dat
...sint2.dat
...sint4.dat
...stack.dat
...tloop.dat
...triloop.dat
...tstackh.dat
...tstacki.dat
...cal_deltaG.py
...featurization.py
...prediction.py
...train_model.py
...LICENSE
...README.md
...test.fasta
```
# Prediction of your own sequence
Then you can construct the imput matrix for prediction

```python
import prediction
prediction.get_score("test.fasta","Cas9",full_length=True,site=None)#You can replace $test.fasta by your own fasta file with 30mer sequence meed the standard of 30mer(4 bp+23 bp+3 bp) you want to predict.
##Then the result file named "type_GNL_result.csv" will show in the same direction of prediction.py file
##If you want the models of specific sites among each editing system please send the email to luoyonglun@genomics.cn.
```
# Output
e.g.
,30mer,GNL-Scorer
0,AGAAGACAACCTATTATCAAAAAAAGGAAA,-0.0032399141226204797
1,GTGCCAGGGCCAAGGTAGGCAAAAAGGAAA,0.0323898160974605
#GNL-ScorerV2 outputs will between 0 and 1, higher value means higher efficiency, which means this gRNA can be more useful in your future applications
```
