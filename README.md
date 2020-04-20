# CRISPR TRAP-seq
This is a ML learning model to predict the gRNA efficiency  of trap-based editing system. eg.ABE, CBE, Cas9

# Environment: Linux

# DeltaG calculation (environmental preparation)

To get the binary file of mfold, please download the package from the website: http://unafold.rna.albany.edu/?q=mfold/download-mfold
```bash
#untar the tar.gz file
tar -zxvf mfold-3.6.tar.gz
#install mfold
./configure prefix="$dir_you_want_to_see_the_bin"
make
make install
#then you can see the bin file in the dir you set in the $prefix
```
Exporting the "quikfold" into your PATH (note:quikfold can help you to calculate the deltaG of multiple sequences at the same time!!! we strongly suggest the user exploit this binary file)

```bash
##Adding the following code into your ~/.bashrc
export PATH=$your_bin_dir_of_mfold:$PATH
```

Before calculating the deltaG, you should prepare a fasta file, which contain 30mer(4 bp+23 bp+3 bp)sequence
note: you should put this file into the working direction under $ABE_CBE_Cas9_trap_ML

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
# Prediction your own sequence
Then you can construct the imput matrix for prediction

```python
import featurization
import prediction
input_matrix=featurization.get_input("test.fasta")#You can replace $test.fasta by your own fasta file with 30mer sequence meed the standard of 30mer(4 bp+23 bp+3 bp) you want to predict.
output=prediction.predict(input_matrix,typ="ABE",full_length=True)
##If you want the models of specific sites among each editing system please send the email to luoyonglun@genomics.cn.
```
# Output

```bash
this output should between 0 and 1, higher value means higher efficiency, and can be more useful in your future application
```
