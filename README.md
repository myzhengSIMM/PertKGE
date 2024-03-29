# Identify compound-protein interaction with knowledge graph embedding of transcriptomic perturbation profiles
This repo contains a PyTorch implementation for PertKGE, which is model proposed in our paper **"Identify compound-protein interaction with knowledge graph embedding of transcriptomic perturbation profiles"**.

## Brief introduction
PertKGE is a method designed to improve compound-protein interaction with knowledge graph embedding of transcriptomic perturbation profiles.The key is to construct a chemical perturbation profiles-based knowledge graph that breaks down genes into DNAs, messenger RNAs (mRNAs), long non-coding RNAs (lncRNAs), microRNAs (miRNAs), transcription factors (TFs), RNA-binding proteins (RBPs) and other protein-coding genes. This enables PertKGE to consider various fine-grained interactions between genes to simulate post-transcriptional and post-translational regulatory events in biological system, which intuitively aligns more closely with real world cellular responses to chemical perturbations.\
![](./Figure1.jpg)

## Requirements
To run our code, please install following main dependency packages
```
python         3.7
torch          1.13.1
torchkge       0.17.5
numpy          1.21.5
pandas         1.1.5
tqdm           4.64.1
cmapPy         4.0.1
```
Others and detailed version can be touched in requirements.txt, conda environment is also recommended

## Demo
We provided demo to reproduce **Target inference scenario I** in our paper.

## Baseline
In this work, we compare to following baseline in different settings
```
# Target Inference
CMap, De, DeMAND, ProTINA, FL-DTD, SSGCN

# Virtual Screening
SSGCN, Glide-SP

# Unbiased test
HetioNet, BioKG, PrimeKG
```
[**CMap**](https://clue.io/) and [**FL-DTD**](http://menglab.pub/fldtd/) were tested in their website. **DeMAND package** was installed through [Bioconductor](https://www.bioconductor.org/packages/release/bioc/html/DeMAND.html). To test **ProTINA**, plz refer to [this repo](https://github.com/CABSEL/ProTINA/tree/master). 
We recommend to test **SSGCN** using [pytorch version](https://github.com/myzhengSIMM/SSGCN).  
  
**Glide-SP** is performed using Maestro of Schrödinger Suites (version 2020-4), and obtained poses were analyzed with PyMOL. [**Grid files**](https://drive.google.com/drive/folders/1wPcn7EaQldWbXONrRVd-ZOcBsNo6IXHw?usp=drive_link) are provided.  

To get other biomedical knowledge graph for comparation, plz refer to [HetioNet](https://github.com/hetio/hetionet/tree/main), [BioKG](https://github.com/dsi-bdi/biokg), [PrimeKG](https://github.com/mims-harvard/PrimeKG). We also provide processed version in Data section.

## Notes
```
3/19      Upload data and model weight for "Secondary pharmacology study of K-756 by PertKGE"
3/20      Upload data and model weight for "PertKGE identified five new scaffold hits for ALDH1B1"
```
FUll Data & Code will coming soon!
