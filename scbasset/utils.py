import tensorflow as tf
import numpy as np
import anndata
from scipy import sparse
import h5py

# call basenji functions from basenji or scbasset
#from basenji.dna_io import dna_1hot
#from basenji.bed import make_bed_seqs
#from basenji.layers import GELU
#from basenji.blocks import conv_block, conv_tower, dense_block, final
from scbasset.basenji_utils import *

###############################
# function for pre-processing #
###############################
def make_h5(input_ad, input_bed, input_fasta, out_file, seq_len=1344, train_ratio=0.9):
    # generate 1-hot-encoding from bed
    seqs_dna, seqs_coords = make_bed_seqs(input_bed, 
                                              fasta_file=input_fasta, 
                                              seq_len=seq_len)
    dna_array = [dna_1hot(x) for x in seqs_dna]
    dna_array = np.array(dna_array)
    ids = np.arange(dna_array.shape[0])
    np.random.seed(10)
    test_val_ids = np.random.choice(ids, int(len(ids)*(1-train_ratio)), replace=False)
    train_ids = np.setdiff1d(ids, test_val_ids)
    val_ids = np.random.choice(test_val_ids, int(len(test_val_ids)/2), replace=False)
    test_ids = np.setdiff1d(test_val_ids, val_ids)
    
    # generate binary peak*cell matrix
    ad = anndata.read_h5ad(input_ad)
    if (sparse.issparse(ad.X)):
        m = (np.array(ad.X.todense()).transpose()!=0)*1
    else:
        m = (np.array(ad.X).transpose()!=0)*1
    
    # save train_test_val splits
    f = h5py.File(out_file, "w")
    f.create_dataset("X", data=dna_array, dtype='bool')
    f.create_dataset("Y", data=m, dtype='int8')
    f.create_dataset("train_ids", data=train_ids, dtype='int')
    f.create_dataset("val_ids", data=val_ids, dtype='int')
    f.create_dataset("test_ids", data=test_ids, dtype='int')
    f.close()

################
# create model #
################
def make_model(bottleneck_size, n_cells, show_summary=True):
    sequence = tf.keras.Input(shape=(1344, 4), name='sequence')
    current = sequence
    current = conv_block(current, filters=288, kernel_size=17, pool_size=3)
    current = conv_tower(current, filters_init=288, filters_mult=1.122, repeat=6, kernel_size=5, pool_size=2)
    current = conv_block(current, filters=256, kernel_size=1)
    current = dense_block(current, flatten=True, units=bottleneck_size, dropout=0.2)
    current = GELU()(current)
    current = final(current, units=n_cells, activation='sigmoid')
    current = tf.keras.layers.Flatten()(current)
    model = tf.keras.Model(inputs=sequence, outputs=current)
    if show_summary:
        model.summary()
    return model


################################
# function for post-processing #
################################

# get cell embeddings
def get_cell_embedding(model):
    return model.layers[-2].get_weights()[0].transpose()

# get the intercept (capture sequenceing depth)
def get_intercept(model):
    return model.layers[-2].get_weights()[1]

# Perform imputation. Didn't normalize for depth.
def imputation_Y(X, model):
    Y_impute = model.predict(X)
    return Y_impute

# perform imputation. Depth normalized.
def imputation_Y_normalize(X, model, scale_method=None):
    
    new_model = tf.keras.Model(inputs=model.layers[0].input, outputs=model.layers[-3].output)
    Y_pred = new_model.predict(X)
    w = model.layers[-2].get_weights()[0]
    accessibility_norm = np.dot(Y_pred.squeeze(), w)
    
    if scale_method=='all_positive':
        accessibility_norm = accessibility_norm - np.min(accessibility_norm)
    if scale_method=='sigmoid':
        accessibility_norm = np.divide(1, 1+np.exp(-accessibility_norm))

    return accessibility_norm