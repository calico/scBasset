import tensorflow as tf
import numpy as np
import pysam
import random
import sys

def dna_1hot(seq, seq_len=None, n_uniform=False):
    """ dna_1hot
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
    if n_uniform:
        seq_code = np.zeros((seq_len, 4), dtype="float16")
    else:
        seq_code = np.zeros((seq_len, 4), dtype="bool")

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == "A":
                seq_code[i, 0] = 1
            elif nt == "C":
                seq_code[i, 1] = 1
            elif nt == "G":
                seq_code[i, 2] = 1
            elif nt == "T":
                seq_code[i, 3] = 1
            else:
                if n_uniform:
                    seq_code[i, :] = 0.25
                else:
                    ni = random.randint(0, 3)
                    seq_code[i, ni] = 1
    return seq_code


# functions from Basenji, can be directly called
def make_bed_seqs(bed_file, fasta_file, seq_len, stranded=False):
    """Return BED regions as sequences and regions as a list of coordinate
    tuples, extended to a specified length."""
    """Extract and extend BED sequences to seq_len."""
    fasta_open = pysam.Fastafile(fasta_file)

    seqs_dna = []
    seqs_coords = []

    for line in open(bed_file):
        a = line.split()
        chrm = a[0]
        start = int(float(a[1]))
        end = int(float(a[2]))
        if len(a) >= 6:
            strand = a[5]
        else:
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


class GELU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)

    def call(self, x):
        # return tf.keras.activations.sigmoid(1.702 * x) * x
        return tf.keras.activations.sigmoid(tf.constant(1.702) * x) * x


def conv_block(
    inputs,
    filters=None,
    kernel_size=1,
    activation="gelu",
    activation_end=None,
    strides=1,
    dilation_rate=1,
    l2_scale=0,
    dropout=0,
    conv_type="standard",
    residual=False,
    pool_size=1,
    batch_norm=True,
    bn_momentum=0.90,
    bn_gamma=None,
    bn_type="standard",
    kernel_initializer="he_normal",
    padding="same",
):
    """Construct a single convolution block.
    Args:
        inputs:        [batch_size, seq_length, features] input sequence
        filters:       Conv1D filters
        kernel_size:   Conv1D kernel_size
        activation:    relu/gelu/etc
        strides:       Conv1D strides
        dilation_rate: Conv1D dilation rate
        l2_scale:      L2 regularization weight.
        dropout:       Dropout rate probability
        conv_type:     Conv1D layer type
        residual:      Residual connection boolean
        pool_size:     Max pool width
        batch_norm:    Apply batch normalization
        bn_momentum:   BatchNorm momentum
        bn_gamma:      BatchNorm gamma (defaults according to residual)
      Returns:
        [batch_size, seq_length, features] output sequence
      """

    # flow through variable current
    current = inputs

    # choose convolution type
    if conv_type == "separable":
        conv_layer = tf.keras.layers.SeparableConv1D
    else:
        conv_layer = tf.keras.layers.Conv1D

    if filters is None:
        filters = inputs.shape[-1]

    # activation
    current = GELU()(current)

    # convolution
    current = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
    )(current)

    # batch norm
    if batch_norm:
        if bn_gamma is None:
            bn_gamma = "zeros" if residual else "ones"
        if bn_type == "sync":
            bn_layer = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            bn_layer = tf.keras.layers.BatchNormalization
        current = bn_layer(momentum=bn_momentum, gamma_initializer=bn_gamma)(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    # residual add
    if residual:
        current = tf.keras.layers.Add()([inputs, current])

    # Pool
    if pool_size > 1:
        current = tf.keras.layers.MaxPool1D(pool_size=pool_size, padding=padding)(
            current
        )

    return current


def conv_tower(
    inputs,
    filters_init,
    filters_end=None,
    filters_mult=None,
    divisible_by=1,
    repeat=1,
    **kwargs
):
    """Construct a reducing convolution block.
    Args:
        inputs:        [batch_size, seq_length, features] input sequence
        filters_init:  Initial Conv1D filters
        filters_end:   End Conv1D filters
        filters_mult:  Multiplier for Conv1D filters
        divisible_by:  Round filters to be divisible by (eg a power of two)
        repeat:        Tower repetitions
    Returns:
        [batch_size, seq_length, features] output sequence
      """

    def _round(x):
        return int(np.round(x / divisible_by) * divisible_by)

    # flow through variable current
    current = inputs

    # initialize filters
    rep_filters = filters_init

    # determine multiplier
    if filters_mult is None:
        assert filters_end is not None
        filters_mult = np.exp(np.log(filters_end / filters_init) / (repeat - 1))

    for ri in range(repeat):
        # convolution
        current = conv_block(current, filters=_round(rep_filters), **kwargs)

        # update filters
        rep_filters *= filters_mult

    return current


def dense_block(
    inputs,
    units=None,
    activation="gelu",
    activation_end=None,
    flatten=False,
    dropout=0,
    l2_scale=0,
    l1_scale=0,
    residual=False,
    batch_norm=True,
    bn_momentum=0.90,
    bn_gamma=None,
    bn_type="standard",
    kernel_initializer="he_normal",
    **kwargs
):
    """Construct a single convolution block.
    Args:
        inputs:         [batch_size, seq_length, features] input sequence
        units:          Conv1D filters
        activation:     relu/gelu/etc
        activation_end: Compute activation after the other operations
        flatten:        Flatten across positional axis
        dropout:        Dropout rate probability
        l2_scale:       L2 regularization weight.
        l1_scale:       L1 regularization weight.
        residual:       Residual connection boolean
        batch_norm:     Apply batch normalization
        bn_momentum:    BatchNorm momentum
        bn_gamma:       BatchNorm gamma (defaults according to residual)
    Returns:
        [batch_size, seq_length(?), features] output sequence
    """
    current = inputs

    if units is None:
        units = inputs.shape[-1]

    # activation
    current = GELU()(current)

    # flatten
    if flatten:
        _, seq_len, seq_depth = current.shape
        current = tf.keras.layers.Reshape((1, seq_len * seq_depth,))(current)

    # dense
    current = tf.keras.layers.Dense(
        units=units,
        use_bias=(not batch_norm),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale),
    )(current)

    # batch norm
    if batch_norm:
        if bn_gamma is None:
            bn_gamma = "zeros" if residual else "ones"
        if bn_type == "sync":
            bn_layer = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            bn_layer = tf.keras.layers.BatchNormalization
        current = bn_layer(momentum=bn_momentum, gamma_initializer=bn_gamma)(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    # residual add
    if residual:
        current = tf.keras.layers.Add()([inputs, current])

    return current


def final(
    inputs,
    units,
    activation="linear",
    flatten=False,
    kernel_initializer="he_normal",
    l2_scale=0,
    l1_scale=0,
    **kwargs
):
    """Final simple transformation before comparison to targets.
    Args:
        inputs:         [batch_size, seq_length, features] input sequence
        units:          Dense units
        activation:     relu/gelu/etc
        flatten:        Flatten positional axis.
        l2_scale:       L2 regularization weight.
        l1_scale:       L1 regularization weight.
    Returns:
        [batch_size, seq_length(?), units] output sequence
    """
    current = inputs

    # flatten
    if flatten:
        _, seq_len, seq_depth = current.shape
        current = tf.keras.layers.Reshape((1, seq_len * seq_depth,))(current)

    # dense
    current = tf.keras.layers.Dense(
        units=units,
        use_bias=True,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale),
    )(current)
    return current
