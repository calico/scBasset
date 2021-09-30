#!/usr/bin/env python

import anndata
import h5py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import pickle
import seaborn as sns
import scipy
import configargparse
import sys
from scbasset.utils import *
from scbasset.basenji_utils import *

def make_parser():
    parser = configargparse.ArgParser(
        description="train scBasset on scATAC data")
    parser.add_argument('--h5', type=str,
                       help='path to h5 file.')
    parser.add_argument('--bottleneck', type=int, default=32,
                       help='size of bottleneck layer. Default to 32')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='batch size. Default to 128')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='learning rate. Default to 0.01')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of epochs to train. Default to 1000.')
    parser.add_argument('--out_path', type=str, default='output',
                       help='Output path. Default to ./output/')
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()
    
    out_dir = args.out_path
    bottleneck_size = args.bottleneck
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    h5_file = args.h5

    f = h5py.File(h5_file, 'r')
    X = f['X'][:].astype('float32')
    Y = f['Y'][:].astype('float32')
    
    train_ids = f['train_ids'][:]
    val_ids = f['val_ids'][:]
    test_ids = f['test_ids'][:]

    X_train = X[train_ids]
    Y_train = Y[train_ids]
    X_val = X[val_ids]
    Y_val = Y[val_ids]

    n_cells = Y.shape[1]

    model = make_model(bottleneck_size, n_cells)

    # combine model
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=0.95,beta_2=0.9995)
    model.compile(loss=loss_fn, optimizer=optimizer,
                  metrics=[tf.keras.metrics.AUC(curve='ROC', multi_label=True),
                           tf.keras.metrics.AUC(curve='PR', multi_label=True)])


    # earlystopping, track train AUC
    filepath = '%s/best_model.h5'%out_dir
    callbacks = [
        tf.keras.callbacks.TensorBoard(out_dir),
        tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, save_weights_only=True, monitor='auc', mode='max'),
        tf.keras.callbacks.EarlyStopping(monitor='auc', min_delta=1e-6, mode='max', patience=50, verbose=1)]
    
    # train the model
    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(X_val, Y_val))
    pickle.dump(history.history, open('%s/history.pickle'%out_dir, 'wb'))
    

if __name__ == "__main__":
    main()
