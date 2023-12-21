# deep_learning_project

##  Project 29 (Group 16 in DTU Learn)

Final project for the course 02456 - Deep Learning, for the Fall Semester 2023.

The main and final version of the scripts developed for this project can be found in the folder `Project29_scripts`.

## Contributors

-   Ana Pastor Mediavilla (s222761)
-   Carlos de Santiago León (s222766)
-   Anu Dinesh Oswal (s222498)
-   Laura Figueiredo Tor (s222797)

# Prediction of protein isoforms using semi-supervised learning

The aim of this project is to create a machine learning algorithm, which was a feed-forward neural network, that can predict isoform expression from gene-expression data obtained from human samples across different tissues. For developing this algorithm, a semi-supervised learning should be implemented to implement a dimensionality reduction of the gene-expression data. Therefore, a VAE algorithm was developed as an unsupervised and deep learning algorithm to perform a feature reduction and to capture informative gene-expression representations that could be beneficial for the isoform predictions.

The supervisors of this project were Jes Frellsen (jefr@dtu.dk) and Kristoffer Vitting-Seerup (krivi@dtu.dk).

## Data

The data used for this project was supplied from the supervisors, and consists on RNA-seq data from different human samples. This RNA-seq data consists mainly on gene-expression levels for different human genes, which is recorded in the units of log2(TPM+1). This measurement refers to Transcripts Per Million, and it is a commonly measurement use when analysing gene expression data.

The data files were suplemented in the HPC server from DTU in the following path: `/dtu-compute/datasets/iso_02456`. As the files were extremely huge, HDF5 files were supported for an eassier way of using the data in the deep learning algorithms implemented. The path in the HPC server for this HDF5 files is the following: `/dtu-compute/datasets/iso 02456/hdf5/`.

Therefore, the data files are not included in this repository (the folders `data` and `hdf5_scripts` only include symbolic links to the paths in the HPC server). For being able to run the scripts of this repository, you will need to clone this GitHub repository in the HPC server for running the codes without any problem.

For this project., a total of 5 files were used:

-   archs4 gene-expression dataset [167883, 18965]: big un-labeled gene expression data used for VAE’s training.
-   gtex gene-expression dataset [17355, 18965]: small paired data containing gene-expression (X for predictive model).
-   gtex isoform-expression dataset [17355, 156958]: small paired data containing isoform-expression (y for predictive model).
-   gtex gene isoform annotation file: gene-isoform relationship for gtex dataset.
-   gtex tissue annotation: tissue types (54) for each sample.

A more detailed explanation of the content and use of these files is included in the report of this project.

## Additional comments

All of the project was developed on the HPC server, as the files were only available on the HPC server. All the code was written in Python scripts and submitted as jobs in the queuing system. For that reason, we were not able to generate a Jupyter Notebook that could recreate the results of the report.
