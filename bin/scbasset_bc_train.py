#!/usr/bin/env python
import anndata
import h5py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import psutil
import math
import pickle
import seaborn as sns
import scipy
import configargparse
import sys
import gc
import pandas as pd
from datetime import datetime
from scbasset.utils import *
from scbasset.basenji_utils import *

def make_parser():
    parser = configargparse.ArgParser(
        description="train scBasset on scATAC data")
    parser.add_argument('--input_folder', type=str,
                       help='folder contains preprocess files. The folder should contain: train_seqs.h5, test_seqs.h5, val_seqs.h5, splits.h5, ad.h5ad')
    parser.add_argument('--bottleneck', type=int, default=32,
                       help='size of bottleneck layer. Default to 32')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='batch size. Default to 128')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='learning rate. Default to 0.01')
    parser.add_argument('--batch_csv', type=str,
                       help='csv file dummy coding for batch information.')
    parser.add_argument('--l2', type=float, default=0,
                       help='regularization on cell embedding path. Increasing this value results in increased mixing. Default to 0. User can try 10e-9, 10e-8, etc.')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of epochs to train. Default to 1000.')
    parser.add_argument('--out_path', type=str, default='output',
                       help='Output path. Default to ./output/')
    parser.add_argument('--print_mem', type=bool, default=True,
                       help='whether to output cpu memory usage.')
    return parser


def main():
    
    parser = make_parser()
    args = parser.parse_args()
    preprocess_folder = args.input_folder
    bottleneck_size = args.bottleneck
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    out_dir = args.out_path
    print_mem = args.print_mem
    l2_reg = args.l2
    batch_file = args.batch_csv
    
    train_data = '%s/train_seqs.h5'%preprocess_folder
    val_data = '%s/val_seqs.h5'%preprocess_folder
    split_file = '%s/splits.h5'%preprocess_folder
    ad = anndata.read_h5ad('%s/ad.h5ad'%preprocess_folder)
    batch_m = pd.read_csv(batch_file, index_col=0)
    n_cells = ad.shape[0]
    
    # csr matrices
    with h5py.File(split_file, 'r') as hf:
        train_ids = hf['train_ids'][:]
        val_ids = hf['val_ids'][:]
    
    m = ad.X.tocoo().transpose().tocsr()
    del ad
    gc.collect()
    
    m_train = m[train_ids,:]
    m_val = m[val_ids,:]
    del m
    gc.collect()
    
    # generate tf.datasets
    train_ds = tf.data.Dataset.from_generator(
        generator(train_data, m_train), 
        output_signature=(
             tf.TensorSpec(shape=(1344,4), dtype=tf.int8),
             tf.TensorSpec(shape=(n_cells), dtype=tf.int8),
        )
    ).shuffle(2000, reshuffle_each_iteration=True).batch(128).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_generator(
        generator(val_data, m_val), 
        output_signature=(
             tf.TensorSpec(shape=(1344,4), dtype=tf.int8),
             tf.TensorSpec(shape=(n_cells), dtype=tf.int8),
        )
    ).batch(128).prefetch(tf.data.AUTOTUNE)
    
    # build model
    model = make_model_bc(bottleneck_size, n_cells, batch_m, l2_1=l2_reg, l2_2=0)

    # compile model
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=0.95,beta_2=0.9995)
    model.compile(loss=loss_fn, optimizer=optimizer,
                  metrics=[tf.keras.metrics.AUC(curve='ROC', multi_label=True),
                           tf.keras.metrics.AUC(curve='PR', multi_label=True)])

    # earlystopping, track train AUC
    filepath = '%s/best_model.h5'%out_dir

    # tensorboard profiler
    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    
    callbacks = [
        tf.keras.callbacks.TensorBoard(out_dir),
        tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, 
                                           save_weights_only=True, monitor='auc', mode='max'),
        tf.keras.callbacks.EarlyStopping(monitor='auc', min_delta=1e-6, 
                                         mode='max', patience=50, verbose=1),
    ]
        
    # train the model
    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds)
    pickle.dump(history.history, open('%s/history.pickle'%out_dir, 'wb'))

if __name__ == "__main__":
    main()
                