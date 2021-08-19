![architecture](https://github.com/calico/scBasset/blob/main/docs/architecture.png)

# scBasset
Sequence-based Modeling of single-cell ATAC-seq using Convolutional Neural Networks.
scBasset is a sequence-based convolutional neural network method to model single cell ATAC-seq data. We show that by leveraging the DNA sequence information underlying accessibility peaks and the expressiveness of a neural network model, scBasset achieves state-of-the-art performance across a variety of tasks on single-cell ATAC-seq and single-cell multiome datasets, including cell type identification, scATAC profile denoising, data integration, and transcription factor activity inference.

## installation
To install scBasset, we suggest first create a conda environment by:
```
    conda env create -n scbasset python=3.7
    conda activate scbasset
```
and then run the following commands:
```
    git clone https://github.com/calico/scBasset.git
    pip install -e scBasset
```
Verify that scBasset is correctly installed by:
```
    import scbasset
```

## Usage
The best way to get familiar with scBasset is to go over the tutorials. Starting from 10x PBMC mulitome output, we will walk you through the data preprocessing, model training and post-processing steps.

### 1. data pre-processing.

See the tutorial here [Preprocess](https://github.com/calico/scBasset/blob/main/examples/preprocess.ipynb). In order to run scBasset model, we need to first pre-process the raw 10x scATAC/multiome output. Two files from 10x scATAC/multiome outputs are required for pre-processing: the **_filtered_feature_bc_matrix.h5** that stores the count matrix, and the **_peaks.bed** file that stores genomic regions of the peaks.
1. We first convert the raw filtered_feature_bc_matrix.h5 into a h5ad file, and perform filtering of peaks. Notice that we filter out peaks accessible in <5% cells for optimal performance.  The h5ad file should have cells as obs and peaks as var. There should be at least three columns in var: chr, start, end that indicate the genomic region of each peak. If the peak list have been filtered, we save the new peak list into a new bed file. Make sure peaks in h5ad file matches peaks in the bed file.
2. make_h5() function from utils.py takes as input the h5ad file, the peak bed file, and the genome fasta file, and output an h5 file that serves as input to the scBasset model.


### 2. training scBasset model.
/bin/train.py is used as a command line tool for training model. Use 'python /bin/train.py --help' to see help page. train.py takes as import an h5 file, which is the output of the data-preprocess.

```
usage: train.py [-h] [--h5 H5] [--bottleneck BOTTLENECK] [--epochs EPOCHS]
                [--out_path OUT_PATH]

train basset on scATAC data

optional arguments:
  -h, --help            show this help message and exit
  --h5 H5               path to h5 file.
  --bottleneck BOTTLENECK
                        size of bottleneck layer. Default to 32
  --epochs EPOCHS       Number of epochs to train. Default to 1000.
  --out_path OUT_PATH   Output path. Default to ./output/
```

### 3. data post-processing.
See the tutorial here [Post-processing](https://github.com/calico/scBasset/blob/main/examples/evaluate.ipynb).


## Basenji
scBasset provides a fixed architecture that we experimented to perform best on sc-ATAC datasets. The key components of scBasset architecture come from [Basenji](https://github.com/calico/basenji). Although scBasset can work as a stand-alone package, we strongly suggest installing [Basenji](https://github.com/calico/basenji) if you want to experiment with alternative archictures. For running training scBasset within using Basenji framework, see [link](link).

## Tutorial: Training scBasset on 10x multiome PBMC dataset (scATAC)
[Preprocess](https://github.com/calico/scBasset/blob/main/examples/preprocess.ipynb)  
[Training](https://github.com/calico/scBasset/blob/main/examples/train.sh)  
[Post-processing](https://github.com/calico/scBasset/blob/main/examples/evaluate.ipynb)  

