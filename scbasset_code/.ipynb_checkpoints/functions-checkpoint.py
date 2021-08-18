import tensorflow as tf
import numpy as np

#############
# functions #
#############

# functions from Basenji
class GELU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)
    def call(self, x):
        # return tf.keras.activations.sigmoid(1.702 * x) * x
        return tf.keras.activations.sigmoid(tf.constant(1.702) * x) * x

def conv_block(inputs, filters=None, kernel_size=1, activation='gelu', activation_end=None,
    strides=1, dilation_rate=1, l2_scale=0, dropout=0, conv_type='standard', residual=False,
    pool_size=1, batch_norm=True, bn_momentum=0.90, bn_gamma=None, bn_type='standard',
    kernel_initializer='he_normal', padding='same'):
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
    if conv_type == 'separable':
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
        padding='same',
        use_bias=False,
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale))(current)

    # batch norm
    if batch_norm:
        if bn_gamma is None:
            bn_gamma = 'zeros' if residual else 'ones'
        if bn_type == 'sync':
            bn_layer = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            bn_layer = tf.keras.layers.BatchNormalization
        current = bn_layer(
            momentum=bn_momentum,
            gamma_initializer=bn_gamma)(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    # residual add
    if residual:
        current = tf.keras.layers.Add()([inputs,current])

    # Pool
    if pool_size > 1:
        current = tf.keras.layers.MaxPool1D(
            pool_size=pool_size,
            padding=padding)(current)

    return current

def conv_tower(inputs, filters_init, filters_end=None, filters_mult=None,
               divisible_by=1, repeat=1, **kwargs):
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
        assert(filters_end is not None)
        filters_mult = np.exp(np.log(filters_end/filters_init) / (repeat-1))

    for ri in range(repeat):
        # convolution
        current = conv_block(current,
                             filters=_round(rep_filters),
                             **kwargs)

        # update filters
        rep_filters *= filters_mult

    return current

def dense_block(inputs, units=None, activation='gelu', activation_end=None,
    flatten=False, dropout=0, l2_scale=0, l1_scale=0, residual=False,
    batch_norm=True, bn_momentum=0.90, bn_gamma=None, bn_type='standard',
    kernel_initializer='he_normal', **kwargs):
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
        current = tf.keras.layers.Reshape((1,seq_len*seq_depth,))(current)

    # dense
    current = tf.keras.layers.Dense(
        units=units,
        use_bias=(not batch_norm),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale)
    )(current)

    # batch norm
    if batch_norm:
        if bn_gamma is None:
            bn_gamma = 'zeros' if residual else 'ones'
        if bn_type == 'sync':
            bn_layer = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            bn_layer = tf.keras.layers.BatchNormalization
        current = bn_layer(
            momentum=bn_momentum,
            gamma_initializer=bn_gamma)(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    # residual add
    if residual:
        current = tf.keras.layers.Add()([inputs,current])

    return current

def final(inputs, units, activation='linear', flatten=False,
          kernel_initializer='he_normal', l2_scale=0, l1_scale=0, **kwargs):
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
        current = tf.keras.layers.Reshape((1,seq_len*seq_depth,))(current)

    # dense
    current = tf.keras.layers.Dense(
        units=units,
        use_bias=True,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale)
    )(current)
    return current


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

def get_cell_embedding(model):
    return model.layers[-2].get_weights()[0].transpose()

def get_intercept(model):
    return model.layers[-2].get_weights()[1]

def imputation_Y(X, model):
    Y_impute = model.predict(X)
    return Y_impute

def imputation_Y_normalize(X, model):
    new_model = tf.keras.Model(inputs=model.layers[0].input, outputs=model.layers[-3].output)
    Y_pred = new_model.predict(X)
    w = model.layers[-2].get_weights()[0]
    accessibility_norm = np.dot(Y_pred.squeeze(), w)
    accessibility_norm = accessibility_norm - np.min(accessibility_norm)
    return accessibility_norm