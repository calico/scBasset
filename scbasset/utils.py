"""
utility functions that support scBasset.
"""

import tensorflow as tf
import numpy as np
import anndata
from scipy import sparse
import h5py
from Bio import SeqIO
from scbasset.basenji_utils import *

###############################
# function for pre-processing #
###############################
def make_h5(
    input_ad,
    input_bed,
    input_fasta,
    out_file,
    seq_len=1344,
    train_ratio=0.9,
):
    """Preprocess to generate h5 for scBasset training.
    Args:
        input_ad:       anndata. the peak by cell matrix.
        input_bed:      bed file. genomic range of peaks.
        input_fasta:    fasta file. genome fasta. (hg19, hg38, mm10 etc.)
        out_file:       output file name.
        seq_len:        peak size to train on. default to 1344.
        train_ratio:    fraction of data used for training. default to 0.9.
    Returns:
        None.           Save a h5 file to 'out_file'. X: 1-hot encoded feature matrix.
                        Y: peak by cell matrix. train_ids: data indices used for train.
                        val_ids: data indices used for val. test_ids: data indices unused
                        during training, can be used for test.
    """
    # generate 1-hot-encoding from bed
    (seqs_dna, seqs_coords,) = make_bed_seqs(
        input_bed,
        fasta_file=input_fasta,
        seq_len=seq_len,
    )
    dna_array = [dna_1hot(x) for x in seqs_dna]
    dna_array = np.array(dna_array)
    ids = np.arange(dna_array.shape[0])
    np.random.seed(10)
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

    # generate binary peak*cell matrix
    ad = anndata.read_h5ad(input_ad)
    if sparse.issparse(ad.X):
        m = (np.array(ad.X.todense()).transpose() != 0) * 1
    else:
        m = (np.array(ad.X).transpose() != 0) * 1

    # save train_test_val splits
    f = h5py.File(out_file, "w")
    f.create_dataset(
        "X",
        data=dna_array,
        dtype="bool",
    )
    f.create_dataset("Y", data=m, dtype="int8")
    f.create_dataset(
        "train_ids",
        data=train_ids,
        dtype="int",
    )
    f.create_dataset(
        "val_ids",
        data=val_ids,
        dtype="int",
    )
    f.create_dataset(
        "test_ids",
        data=test_ids,
        dtype="int",
    )
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


################################
# function for post-processing #
################################
def get_cell_embedding(model):
    """get cell embeddings from trained model"""
    return model.layers[-3].get_weights()[0].transpose()


def get_intercept(model):
    """get intercept from trained model"""
    return model.layers[-3].get_weights()[1]


def imputation_Y(X, model):
    """Perform imputation. Don't normalize for depth.
    Args:
        X:              feature matrix from h5.
        model:          a trained scBasset model.
    Returns:
        array:          a peak*cell imputed accessibility matrix. Sequencing depth
                        isn't corrected for.
    """
    Y_impute = model.predict(X)
    return Y_impute


# perform imputation. Depth normalized.
def imputation_Y_normalize(X, model, scale_method=None):
    """Perform imputation. Normalize for depth.
    Args:
        X:              feature matrix from h5.
        model:          a trained scBasset model.
    Returns:
        array:          a peak*cell imputed accessibility matrix. Sequencing depth corrected
                        for. scale_method=None, don't do any scaling of output. The raw
                        normalized output would have both positive and negative values.
                        scale_method="all_positive" scales the output by subtracting minimum value.
                        scale_method="sigmoid" scales the output by sigmoid transform.
    """
    new_model = tf.keras.Model(
        inputs=model.layers[0].input,
        outputs=model.layers[-4].output,
    )
    Y_pred = new_model.predict(X)
    w = model.layers[-3].get_weights()[0]
    accessibility_norm = np.dot(Y_pred.squeeze(), w)

    if scale_method == "all_positive":
        accessibility_norm = accessibility_norm - np.min(accessibility_norm)
    if scale_method == "sigmoid":
        accessibility_norm = np.divide(
            1,
            1 + np.exp(-accessibility_norm),
        )

    return accessibility_norm


def pred_on_fasta(fa, model):
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
    pred = imputation_Y_normalize(seqs_1hot, model)
    return pred


def motif_score(tf, model, motif_fasta_folder):
    """score motifs for any given TF.
    Args:
        tf:             TF of interest. By default we only score TFs in the
                        scBasset/data/Homo_sapiens_motif_fasta/shuffled_peaks_motifs/ folder.
                        To score on additional motifs. Follow scBasset/examples/make_fasta.R
                        to create dinucleotide shuffled sequences with and without motifs of
                        interest.
        model:          a trained scBasset model.
        motif_fasta_folder: folder for dinucleotide shuffled sequences with and without any motif.
                        We provided motifs from CIS-BP/Homo_sapiens.meme downloaded from the
                        MEME Suite (https://meme-suite.org/meme/) at
                        'scBasset/data/Homo_sapiens_motif_fasta'.
    Returns:
        array:          a vector for motif activity per cell. (cell order is the
                        same order as the model.)
    """
    fasta_motif = "%s/shuffled_peaks_motifs/%s.fasta" % (motif_fasta_folder, tf)
    fasta_bg = "%s/shuffled_peaks.fasta" % motif_fasta_folder

    pred_motif = pred_on_fasta(fasta_motif, model)
    pred_bg = pred_on_fasta(fasta_bg, model)
    tf_score = pred_motif.mean(axis=0) - pred_bg.mean(axis=0)
    tf_score = (tf_score - tf_score.mean()) / tf_score.std()
    return tf_score
