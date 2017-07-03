LEMP RELEASE NOTES
============================

LEMP (LSTM-based Ensemble Malonylation Predictor) Version 1.0

By Xuhan Liu and Zhen Chen, July 1st 2017

Here are some very brief notes on using the LEMP V1.0 software.

Please see the LICENSE file for the license terms for the software.
Basically it's free to academic users.
If you do wish to sell the software or use it in a commercial product,
then please contact xuhanliu@qq.com.

Before compile LEMP, please make sure that integrity of the software.

INSTALLATION
============
Firstly, ensure that the version of your Python >= 2.7.

Secondly, all the following packages are installed in your machine:

    1. Numpy:

        tcsh% pip install numpy

    2. Scikit-learn

        tcsh% pip install scikit-learn

    3. H5py

        tcsh% pip install h5py

    4. Tensorflow

        tcsh% pip install tensorflow (CPU version)
        tcsh% pip install tensorflow_gpu (GPU version)

    5. Keras

        tcsh% pip install keras


Finally decompressing the software.

    tcsh% tar -zxvf lemp-1.0.tar.gz

The executables will be placed in the this directory.

USAGE
============
If you want to predict your own protein sequence, execute as follows:

    python lemp.py  -i <input_file> [-o <output_file>] [-h] [-v]

If you want to rebuild the new model with your own dataset,

    1). Construct your dataset that has the same format as ours in 'dataset/chen_train.txt' and 'dataset/chen_test.txt',
        which is training set and independent set, respectively.

    2). excute following shell:

            python model.py

ARGUMENTS
============
    -i <input_file> : dataset file containing protein sequences as FASTA file format.
    -o <output_dir> : a directory containing the results of prediction of each sample.

    -v              : version information of this software.

    -h              : Help information, print USAGE and ARGUMENTS messages.

Note: Please designate each protein sequence in FASTA file with distinct name!

OUTPUT FORMAT
============
<Seq_ID>   <Site>  <Residue>   <Score>  <Y/N(sp=90%)>   <Y/N(sp=95%)>   <Y/N(sp=99%)>
    1. Seq ID       : The ID of protein sequence, which is as same as in the fasta file.
    2. Site         : The position of the residue at the protein sequence.
    3. Residue      : The type of residue which is possibly malonylated. In common, it must be lysine (i.e. "K").
    4. Score        : The score that computed by the LEMP.
    5. Y/N(sp=90%)  : Judging whether this residue was malonylation site or not under the specification = 90%.
    5. Y/N(sp=95%)  : Judging whether this residue was malonylation site or not under the specification = 95%.
    5. Y/N(sp=99%)  : Judging whether this residue was malonylation site or not under the specification = 99%.

Note: The larger value of specification under which it was judged as malonylation site,
      The higher confidence of the result will be.

Finished.
