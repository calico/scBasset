"""
utility functions that support scBasset.
"""

import anndata
import h5py
import time
import os
import psutil
import numpy as np
import tensorflow as tf
from Bio import SeqIO
from scipy import sparse
from scbasset.basenji_utils import *

###############################
# function for pre-processing #
###############################

def make_bed_seqs_from_df(input_bed, fasta_file, seq_len, stranded=False):
    """Return BED regions as sequences and regions as a list of coordinate
    tuples, extended to a specified length."""
    """Extract and extend BED sequences to seq_len."""
    fasta_open = pysam.Fastafile(fasta_file)

    seqs_dna = []
    seqs_coords = []

    for i in range(input_bed.shape[0]):
        chrm = input_bed.iloc[i,0]
        start = int(input_bed.iloc[i,1])
        end = int(input_bed.iloc[i,2])
        strand = "+"

        # determine sequence limits
        mid = (start + end) // 2
        seq_start = mid - seq_len // 2
        seq_end = seq_start + seq_len

        # save
        if stranded:
            seqs_coords.append((chrm, seq_start, seq_end, strand))
        else:
            seqs_coords.append((chrm, seq_start, seq_end))
        # initialize sequence
        seq_dna = ""
        # add N's for left over reach
        if seq_start < 0:
            print(
                "Adding %d Ns to %s:%d-%s" % (-seq_start, chrm, start, end),
                file=sys.stderr,
            )
            seq_dna = "N" * (-seq_start)
            seq_start = 0

        # get dna
        seq_dna += fasta_open.fetch(chrm, seq_start, seq_end).upper()

        # add N's for right over reach
        if len(seq_dna) < seq_len:
            print(
                "Adding %d Ns to %s:%d-%s" % (seq_len - len(seq_dna), chrm, start, end),
                file=sys.stderr,
            )
            seq_dna += "N" * (seq_len - len(seq_dna))
        # append
        seqs_dna.append(seq_dna)
    fasta_open.close()
    return seqs_dna, seqs_coords


def dna_1hot_2vec(seq, seq_len=None):
    """dna_1hot
    Args:
      seq:       nucleotide sequence.
      seq_len:   length to extend/trim sequences to.
      n_uniform: represent N's as 0.25, forcing float16,
                 rather than sampling.
    Returns:
      seq_code: length by nucleotides array representation.
    """
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim : seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2
    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    seq_code = np.zeros((seq_len, ), dtype="int8")

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == "A":
                seq_code[i] = 0
            elif nt == "C":
                seq_code[i] = 1
            elif nt == "G":
                seq_code[i] = 2
            elif nt == "T":
                seq_code[i] = 3
            else:
                seq_code[i] =  random.randint(0, 3)
    return seq_code

def split_train_test_val(ids, seed=10, train_ratio=0.9):
    np.random.seed(seed)
    test_val_ids = np.random.choice(
        ids,
        int(len(ids) * (1 - train_ratio)),
        replace=False,
    )
    train_ids = np.setdiff1d(ids, test_val_ids)
    val_ids = np.random.choice(
        test_val_ids,
        int(len(test_val_ids) / 2),
        replace=False,
    )
    test_ids = np.setdiff1d(test_val_ids, val_ids)
    return train_ids, test_ids, val_ids
    

def make_h5_sparse(tmp_ad, h5_name, input_fasta, seq_len=1344, batch_size=1000):
    ## batch_size: how many peaks to process at a time
    ## tmp_ad.var must have columns chr, start, end
    
    t0 = time.time()
    
    m = tmp_ad.X
    m = m.tocoo().transpose().tocsr()
    n_peaks = tmp_ad.shape[1]
    bed_df = tmp_ad.var.loc[:,['chr','start','end']] # bed file
    bed_df.index = np.arange(bed_df.shape[0])
    n_batch = int(np.floor(n_peaks/batch_size))
    batches = np.array_split(np.arange(n_peaks), n_batch) # split all peaks to process in batches
    
    ### create h5 file
    # X is a matrix of n_peaks * 1344
    f = h5py.File(h5_name, "w")
    
    ds_X = f.create_dataset(
        "X",
        (n_peaks, seq_len),
        dtype="int8",
    )

    # save to h5 file
    for i in range(len(batches)):
        
        idx = batches[i]
        # write X to h5 file
        seqs_dna,_ = make_bed_seqs_from_df(
            bed_df.iloc[idx,:],
            fasta_file=input_fasta,
            seq_len=seq_len,
        )
        dna_array_dense = [dna_1hot_2vec(x) for x in seqs_dna]
        dna_array_dense = np.array(dna_array_dense)
        ds_X[idx] = dna_array_dense
            
        t1 = time.time()
        total = t1-t0
        print('process %d peaks takes %.1f s' %(i*batch_size, total))
    
    f.close()

################
# create model #
################
def make_model(
    bottleneck_size,
    n_cells,
    seq_len=1344,
    show_summary=True,
):
    """create keras CNN model.
    Args:
        bottleneck_size:int. size of the bottleneck layer.
        n_cells:        int. number of cells in the dataset. Defined the number of tasks.
        seq_len:        int. peak size used to train. Default to 1344.
        show_summary:   logical. Whether to print the model summary. Default to True.
    Returns:
        keras model:    object of keras model class.
    """
    sequence = tf.keras.Input(
        shape=(seq_len, 4),
        name="sequence",
    )
    current = sequence
    (current, reverse_bool,) = StochasticReverseComplement()(
        current
    )  # enable random rv
    current = StochasticShift(3)(current)  # enable random shift
    current = conv_block(
        current,
        filters=288,
        kernel_size=17,
        pool_size=3,
    )  # 1 cnn block
    current = conv_tower(
        current,
        filters_init=288,
        filters_mult=1.122,
        repeat=6,
        kernel_size=5,
        pool_size=2,
    )  # cnn tower
    current = conv_block(
        current,
        filters=256,
        kernel_size=1,
    )
    current = dense_block(
        current,
        flatten=True,
        units=bottleneck_size,
        dropout=0.2,
    )
    current = GELU()(current)
    current = final(
        current,
        units=n_cells,
        activation="sigmoid",
    )
    current = SwitchReverse()(
        [current, reverse_bool]
    )  # switch back, # this doesn't matter
    current = tf.keras.layers.Flatten()(current)
    model = tf.keras.Model(inputs=sequence, outputs=current)
    if show_summary:
        model.summary()
    return model



def make_model_bc(
    bottleneck_size,
    n_cells,
    batch_m,
    l2_1=0,
    l2_2=0,
    seq_len=1344,
    show_summary=True,
):
    """create keras CNN model.
    Args:
        bottleneck_size:int. size of the bottleneck layer.
        n_cells:        int. number of cells in the dataset. Defined the number of tasks.
        batch_m:        pandas DataFrame. dummy-coded binary matrix for batch information.
        seq_len:        int. peak size used to train. Default to 1344.
        show_summary:   logical. Whether to print the model summary. Default to True.
    Returns:
        keras model:    object of keras model class.
    """
        
    sequence = tf.keras.Input(
        shape=(seq_len, 4),
        name="sequence",
    )
    current = sequence
    (current, reverse_bool,) = StochasticReverseComplement()(
        current
    )  # enable random rv
    current = StochasticShift(3)(current)  # enable random shift
    current = conv_block(
        current,
        filters=288,
        kernel_size=17,
        pool_size=3,
    )  # 1 cnn block
    current = conv_tower(
        current,
        filters_init=288,
        filters_mult=1.122,
        repeat=6,
        kernel_size=5,
        pool_size=2,
    )  # cnn tower
    current = conv_block(
        current,
        filters=256,
        kernel_size=1,
    )
    current = dense_block(
        current,
        flatten=True,
        units=bottleneck_size,
        dropout=0.2,
    )
    current = GELU()(current)
    
    # replace the following:
    #current = final(
    #    current,
    #    units=n_cells,
    #    activation="sigmoid",
    #)
    batch_info = tf.constant(batch_m.values.transpose(), dtype='float32') # batch matrix
    current1 = tf.keras.layers.Dense(units=n_cells, # path1
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_1))(current)

    current2 = tf.keras.layers.Dense(units=batch_info.shape[0], # path2
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_2))(current)
    current2 = tf.linalg.matmul(current2, batch_info) 
    current = tf.math.add(current1, current2) # sum
    current = tf.keras.layers.Activation(activation='sigmoid')(current)
    
    # the same as before
    current = SwitchReverse()(
        [current, reverse_bool]
    )  # switch back, # this doesn't matter
    current = tf.keras.layers.Flatten()(current)
    model = tf.keras.Model(inputs=sequence, outputs=current)
    if show_summary:
        model.summary()
    return model


def print_memory():
    process = psutil.Process(os.getpid())
    print('cpu memory used: %.1fGB.'%(process.memory_info().rss/1e9))


# a generator to read examples from h5 file
# create a tf dataset
class generator:
    def __init__(self, file, m):
        self.file = file # h5 file for sequence
        self.m = m # csr matrix, rows as seqs, cols are cells
        self.n_cells = m.shape[1]
        self.ones = np.ones(1344)
        self.rows = np.arange(1344)

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            X = hf['X']
            for i in range(X.shape[0]):
                x = X[i]
                x_tf = sparse.coo_matrix((self.ones, (self.rows, x)), 
                                               shape=(1344, 4), 
                                               dtype='int8').toarray()
                y = self.m.indices[self.m.indptr[i]:self.m.indptr[i+1]]
                y_tf = np.zeros(self.n_cells, dtype='int8')
                y_tf[y] = 1
                yield x_tf, y_tf
                
################################
# function for post-processing #
################################
def get_cell_embedding(model, bc_model=False):
    """get cell embeddings from trained model"""
    if bc_model:
         output = model.layers[-6].get_weights()[0].transpose()
    else:
         output = model.layers[-3].get_weights()[0].transpose()
    return output

def get_intercept(model, bc_model=False):
    """get intercept from trained model"""
    if bc_model:
        output = model.layers[-6].get_weights()[1]    
    else:
        output = model.layers[-3].get_weights()[1]
    return output


def imputation_Y(X, model, bc_model=False):
    """Perform imputation. Don't normalize for depth.
    Args:
        X:              feature matrix from h5.
        model:          a trained scBasset model.
    Returns:
        array:          a peak*cell imputed accessibility matrix. Sequencing depth
                        isn't corrected for.
    """
    if bc_model:
        new_model = tf.keras.Model(
            inputs=model.layers[0].input,
            outputs=model.layers[-6].output
        )
        Y_impute = new_model.predict(X)
    else:
        Y_impute = model.predict(X)
    return Y_impute


# perform imputation. Depth normalized.
def imputation_Y_normalize(X, model, bc_model=False, scale_method='sigmoid'):
    """Perform imputation. Normalize for depth.
    Args:
        X:              feature matrix from h5.
        model:          a trained scBasset model.
    Returns:
        array:          a peak*cell imputed accessibility matrix. Sequencing depth corrected
                        for. scale_method=None, don't do any scaling of output. The raw
                        normalized output would have both positive and negative values.
                        scale_method="positive" scales the output by subtracting minimum value.
                        scale_method="sigmoid" scales the output by sigmoid transform.
    """
    if bc_model:
        new_model = tf.keras.Model(
            inputs=model.layers[0].input,
            outputs=model.layers[-8].output,
        )
        Y_pred = new_model.predict(X)
        w = model.layers[-6].get_weights()[0]
        intercepts = model.layers[-6].get_weights()[1]
        accessibility_norm = np.dot(Y_pred.squeeze(), w)
    
    else:
        new_model = tf.keras.Model(
            inputs=model.layers[0].input,
            outputs=model.layers[-4].output,
        )
        Y_pred = new_model.predict(X)
        w = model.layers[-3].get_weights()[0]
        intercepts = model.layers[-3].get_weights()[1]
        accessibility_norm = np.dot(Y_pred.squeeze(), w)

    # scaling
    if scale_method == "positive":
        accessibility_norm = accessibility_norm - np.min(accessibility_norm)
    
    if scale_method == "sigmoid":
        #median_depth = np.median(intercepts)
        norm_depth = 0
        accessibility_norm = np.divide(
            1,
            1 + np.exp(-(accessibility_norm+norm_depth)))
    
    return accessibility_norm


def pred_on_fasta(fa, model, bc=False, scale_method='sigmoid'):
    """Run a trained model on a fasta file.
    Args:
        fa:             fasta file to run on. Need to have a fixed size of 1344. Default
                        sequence size of trained model.
        model:          a trained scBasset model.
    Returns:
        array:          a peak*cell imputed accessibility matrix. Sequencing depth corrected for.
    """
    records = list(SeqIO.parse(fa, "fasta"))
    seqs = [str(i.seq) for i in records]
    seqs_1hot = np.array([dna_1hot(i) for i in seqs])
    pred = imputation_Y_normalize(seqs_1hot, model, bc_model=bc, scale_method=scale_method)
    return pred


def motif_score(tf, model, motif_fasta_folder, bc=False, scale_method='sigmoid'):
    """score motifs for any given TF.
    Args:
        tf:             TF of interest. By default we only provide TFs to score in
                        https://storage.googleapis.com/scbasset_tutorial_data/Homo_sapiens_motif_fasta.tar.gz.
                        To score on additional motifs, follow make_fasta.R in the tarball 
                        to create dinucleotide shuffled sequences with and without motifs of
                        interest.
        model:          a trained scBasset model.
        motif_fasta_folder: folder for dinucleotide shuffled sequences with and without any motif.
                        We provided motifs from CIS-BP/Homo_sapiens.meme downloaded from the
                        MEME Suite (https://meme-suite.org/meme/) in 
                        https://storage.googleapis.com/scbasset_tutorial_data/Homo_sapiens_motif_fasta.tar.gz.
    Returns:
        array:          a vector for motif activity per cell. (cell order is the
                        same order as the model.)
    """
    fasta_motif = "%s/shuffled_peaks_motifs/%s.fasta" % (motif_fasta_folder, tf)
    fasta_bg = "%s/shuffled_peaks.fasta" % motif_fasta_folder

    pred_motif = pred_on_fasta(fasta_motif, model, bc=bc, scale_method='sigmoid')
    pred_bg = pred_on_fasta(fasta_bg, model, bc=bc, scale_method='sigmoid')
    tf_score = pred_motif.mean(axis=0) - pred_bg.mean(axis=0)
    tf_score = (tf_score - tf_score.mean()) / tf_score.std()
    return tf_score

# compute ism from sequence
def ism(seq_ref_1hot, model):
    
    new_model = tf.keras.Model(
        inputs=model.layers[0].input,
        outputs=model.layers[-4].output,
    )
    w = model.layers[-3].get_weights()[0]

    # output matrix
    m = np.zeros((model.output.shape[1], seq_ref_1hot.shape[0], seq_ref_1hot.shape[1]))
    
    # predication of reference seq
    seqs_1hot_tf = tf.convert_to_tensor(seq_ref_1hot, dtype=tf.float32)[tf.newaxis]
    latent_ref = new_model(seqs_1hot_tf)
    latent_ref = tf.squeeze(latent_ref, axis=[0,1])

    # compute ism
    for i in range(seq_ref_1hot.shape[0]):
        out = []
        for j in range(4):
            tmp = np.copy(seq_ref_1hot)
            tmp[i,:] = [False, False, False, False]
            tmp[i,j] = True
            out += [tmp]
        
        out_tf = tf.convert_to_tensor(np.array(out), dtype=tf.float32)
        latent = new_model(out_tf)
        latent = tf.squeeze(latent, axis=[1])
        latent = latent - latent_ref # ism on latent space

        pred = tf.einsum('nb,bt->nt', latent, w)
        m[:,i,:] = pred.numpy().transpose()
        
    return m