<img src="https://github.com/calico/scBasset/blob/main/docs/architecture.png" width=100% height=100%>

<!---
![architecture](https://github.com/calico/scBasset/blob/main/docs/architecture.png)
-->

# scBasset
Sequence-based Modeling of single-cell ATAC-seq using Convolutional Neural Networks.
scBasset is a sequence-based convolutional neural network method to model single cell ATAC-seq data. We show that by leveraging the DNA sequence information underlying accessibility peaks and the expressiveness of a neural network model, scBasset achieves state-of-the-art performance across a variety of tasks on single-cell ATAC-seq and single-cell multiome datasets, including cell type identification, scATAC profile denoising, data integration, and transcription factor activity inference.

## installation
To install scBasset, we suggest first create a conda environment by:
```
    conda create -n scbasset python=3.7
    conda activate scbasset
```
and then run the following commands:
```
    git clone https://github.com/calico/scBasset.git
    pip install -e scBasset
```
Installation should take only a few minutes. Verify that scBasset is correctly installed by running in python:
```
    import scbasset
```

## Usage
The best way to get familiar with scBasset is to go over the tutorials. Starting from 10x PBMC mulitome output, we will walk you through the data preprocessing, model training and post-processing steps.

## Tutorial: Training scBasset on 10x multiome PBMC dataset (scATAC)

[Download tutorial data](https://github.com/calico/scBasset/blob/main/examples/download.ipynb)   
[Create Anndata](https://github.com/calico/scBasset/blob/main/examples/PBMC_multiome/make_anndata.ipynb)   
[create H5](https://github.com/calico/scBasset/blob/main/examples/PBMC_multiome/train.sh)   
[Training](https://github.com/calico/scBasset/blob/main/examples/PBMC_multiome/train.sh)   
[Get cell embeddings](https://github.com/calico/scBasset/blob/main/examples/PBMC_multiome/evaluate.ipynb)   
[Motif scoring](https://github.com/calico/scBasset/blob/main/examples/PBMC_multiome/score_motif.ipynb)  
[ISM_visualization](https://github.com/calico/scBasset/blob/main/examples/ISM/ism.ipynb)   
[ISM_PWM](https://github.com/calico/scBasset/blob/main/examples/ISM/motif_analysis.ipynb)    
[batch correction tutorial](https://github.com/calico/scBasset/blob/main/examples/batch_correction/buen_batch_correction.ipynb)    
 
### 1. download tutorial data.

Follow [Download tutorial data](https://github.com/calico/scBasset/blob/main/examples/download.ipynb) to download data used for tutorial.

### 2. create anndata.

See the tutorial here [Create Anndata](https://github.com/calico/scBasset/blob/main/examples/PBMC_multiome/make_anndata.ipynb). 
In order to run scBasset model, we need to first create anndata from the raw 10x scATAC/multiome output. Two files from 10x scATAC/multiome outputs are required: the **_filtered_feature_bc_matrix.h5** that stores the count matrix, and the **_peaks.bed** file that stores genomic regions of the peaks. Briefly, We converted the raw filtered_feature_bc_matrix.h5 into a h5ad file, and perform filtering of peaks. Notice that we filter out peaks accessible in <5% cells for optimal performance.  The h5ad file should have cells as obs and peaks as var. There should be at least three columns in var: chr, start, end that indicate the genomic region of each peak.

### 3. generate training data for scBasset.

scBasset/bin/scbasset_preprocess.py is used as a command line tool to extract sequences underlying the peaks, one-hot encode them and save them to h5 sparse format. Run 'scbasset_preprocess.py --help' to see help page. scBasset takes as input anndata file, genome fasta file for the corresponding genome build, and output folder name.  Genome fasta file can be downloaded from [UCSC](https://hgdownload.soe.ucsc.edu/downloads.html). This process should take ~30s for example multiome PBMC dataset.
The following output are saved to the output folder: ad.h5ad, splits.h5, train_seqs.h5, test_seqs.h5, val_seqs.h5.
```
usage: scbasset_preprocess.py [-h] [--ad_file AD_FILE]
                              [--input_fasta INPUT_FASTA]
                              [--out_path OUT_PATH]

Preprocess anndata to generate inputs for scBasset.

optional arguments:
  -h, --help            show this help message and exit
  --ad_file AD_FILE     Input scATAC anndata. .var must have 'chr', 'start', 'end' columns. anndata.X must be in csr format.
  --input_fasta INPUT_FASTA
                        Genome fasta file.
  --out_path OUT_PATH   Output path. Default to ./processed/
```
Note: if you are generating peak atlas and count matrix from other sources such as ArchR. Make sure the peak atlas and count matrix have matching peak order! For ArchR, use rowRanges(getMatrixFromProject(proj,"PeakMatrix")) as the peaks so that the peak order matches getMatrixFromProject(proj, "PeakMatrix"). Peaks in getPeakSet(proj) could have a different order!

### 4. train scBasset model.
scBasset/bin/scbasset_train.py is used as a command line tool for training model. Use 'scbasset_train.py --help' to see help page. scbasset_train.py takes as input the folder path containing preprocessed h5 files. scBasset by default trains for 1000 epochs with early-stopping on **train auc**. We focused on **training auROC** instead of validation auROC for model selection because we observed that, across multiple datasets, the model continues to improve cell embeddings even after the point where the validation auROC has plateaued. We observed that validation auROC during the later stages of training is stable, suggesting that the model is not prone to overfitting. Our analyses indicate that the 32 units bottleneck layer is a major impediment to true overfitting. Training takes ~13s per epoch for example multiome PBMC dataset on V100 gpu. 
```
usage: scbasset_train.py [-h] [--input_folder INPUT_FOLDER]
                         [--bottleneck BOTTLENECK] [--batch_size BATCH_SIZE]
                         [--lr LR] [--epochs EPOCHS] [--out_path OUT_PATH]
                         [--print_mem PRINT_MEM]

train scBasset on scATAC data

optional arguments:
  -h, --help            show this help message and exit
  --input_folder INPUT_FOLDER
                        folder contains preprocess files. The folder should
                        contain: train_seqs.h5, test_seqs.h5, val_seqs.h5,
                        splits.h5, ad.h5ad
  --bottleneck BOTTLENECK
                        size of bottleneck layer. Default to 32
  --batch_size BATCH_SIZE
                        batch size. Default to 128
  --lr LR               learning rate. Default to 0.01
  --epochs EPOCHS       Number of epochs to train. Default to 1000.
  --out_path OUT_PATH   Output path. Default to ./output/
  --print_mem PRINT_MEM
                        whether to output cpu memory usage.
```

### 5. use trained model for downstream analysis.
See the tutorial [Get cell embeddings](https://github.com/calico/scBasset/blob/main/examples/PBMC_multiome/evaluate.ipynb) for how to get cell embedding and denoised accessibility profiles from a trained scBasset model.  

See the tutorial [Motif scoring](https://github.com/calico/scBasset/blob/main/examples/PBMC_multiome/score_motif.ipynb) for how to score motifs on a per cell basis using motif injection method. For motif injection, we first generated dinucleotides shuffled background sequences, and inserted motif of interest to the center of those sequences. We provided such sequences for motifs in the MEME Suite CIS-BP 1.0 [Homo sapiens motif collection](https://meme-suite.org/meme/db/motifs) at [Homo_sapiens_motif_fasta](https://storage.googleapis.com/scbasset_tutorial_data/Homo_sapiens_motif_fasta.tar.gz). To score on additional motifs, follow make_fasta.R in the tarball to create dinucleotide shuffled sequences with and without motifs of interest. 

See the tutorial [ISM_visualization](https://github.com/calico/scBasset/blob/main/examples/ISM/ism.ipynb) for performing in silico saturation mutagenesis on an example peak of interest, and visualizing the ISM scores aggregated by cell type. See tutorial [ISM_PWM](https://github.com/calico/scBasset/blob/main/examples/ISM/motif_analysis.ipynb) for computing Pearson correlation between ISM and motif match.

### 6. batch correction.

scBasset can be adapted to perform batch correction. The /bin/scbasset_bc_train.py commandline tool is for training scBasset with batch correction. See [batch correction tutorial](https://github.com/calico/scBasset/blob/main/examples/batch_correction/buen_batch_correction.ipynb) for an example of correcting for donor effect on Buenrostro2018 dataset.  We use a parameter l2 to control for the level of mixing. Usually a default l2 value of 0 gives good mixing performance. In cases where default l2 doesn't mix enough, you can try tune up l2 (1e-9, 1e-8 etc).

## Basenji
scBasset provides a fixed architecture that we experimented to perform best on scATAC datasets. The key components of scBasset architecture come from [Basenji](https://github.com/calico/basenji). Although scBasset can work as a stand-alone package, we strongly suggest installing [Basenji](https://github.com/calico/basenji) if you want to experiment with alternative archictures. See [link](link) as an example of how to create a Json file and instruct Basenji to train a scBasset model.
