Noise Contrastive Estimation Projected Linear Recommender(NCE-PLRec)
====================================================================
![](https://img.shields.io/badge/linux-ubuntu-red.svg)
![](https://img.shields.io/badge/Mac-OS-red.svg)

![](https://img.shields.io/badge/cuda-8.0-green.svg)
![](https://img.shields.io/badge/python-2.7-green.svg)

![](https://img.shields.io/badge/cython-0.28.5-blue.svg)
![](https://img.shields.io/badge/cupy-4.0.0-blue.svg)
![](https://img.shields.io/badge/scipy-1.0.0-blue.svg)
![](https://img.shields.io/badge/numpy-1.14.1-blue.svg)
![](https://img.shields.io/badge/sklearn-0.19.0-blue.svg)
![](https://img.shields.io/badge/pandas-0.20.3-blue.svg)
![](https://img.shields.io/badge/tqdm-4.11.2-blue.svg)
![](https://img.shields.io/badge/argparse-1.1-blue.svg)
![](https://img.shields.io/badge/tensorflow-1.4.0-blue.svg)
![](https://img.shields.io/badge/pytorch-1.0.0-blue.svg)
![](https://img.shields.io/badge/matplotlib-3.0.0-blue.svg)
![](https://img.shields.io/badge/fbpca-1.0-blue.svg)
![](https://img.shields.io/badge/pyyaml-4.1-blue.svg)


If you are interested in building up your research on this work, please cite:
```
@inproceedings{sigir19a,
  author    = {Ga Wu and Maksims Volkovs and Chee Loong Soon and Scott Sanner and Himanshu Rai},
  title     = {Noise Contrastive Estimation for One-Class Collaborative Filtering},
  booktitle = {Proceedings of the 42nd International {ACM} {SIGIR} Conference on Research and Development in Information Retrieval {(SIGIR-19)}},
  address   = {Paris, France},
  year      = {2019}
}
```

# Author Affiliate
<p align="center">
<a href="https://www.utoronto.ca//"><img src="https://github.com/wuga214/NCE_Projected_LRec/blob/master/logos/U-of-T-logo.svg" width="140"></a> | 
<a href="https://layer6.ai/"><img src="https://github.com/wuga214/NCE_Projected_LRec/blob/master/logos/logo.svg" width="120"></a> | 
<a href="https://vectorinstitute.ai/"><img src="https://github.com/wuga214/NCE_Projected_LRec/blob/master/logos/vectorlogo.svg" width="140"></a>
</p>



# Algorithm Implemented
1. Noise Contrastive Estimation Projected Linear Recommender(NCE-PLRec)
2. Variational Autoencoder for Collaborative Filtering(VAE-CF)
3. Collaborative Metric Learning(CML)
4. Auto-encoder Recommender(AutoRec)
5. Collaborative Denoising Auto-Encoders(CDAE)
6. Weighted Regularized Matrix Factorization(WRMF)
7. Pure SVD Recommender(PureSVD)
8. Bayesian Personalized Ranking(BPR)
9. Noise Contrastive Estimation through SVD(NCE-SVD)
10. Popularity


# Data
1. Movielens 1M,
2. Movielens 20M,
3. Yahoo 1R,
4. Netflix,
5. Amazon Digital Music (2018),
6. Amazon Video Games (2018)

Experiment result for the validation datasets are listed under `tables` folder. The experiments are conducted on Compute Canada cluster with multiple runs with all valid hyper-parameter combinations. Please use those as benchmark reference if needed.

Data is not suit to submit on github, so please prepare it yourself. It should be numpy npy file directly
dumped from csr sparse matrix. It should be easy..

# Measure
The above algorithm could be splitted into two major category based on the distance
measurement: Euclidean or Cosine. CML is a euclidean distance recommender. And, ALS
is a typical Cosine distance recommender. When doing evaluation, please select
similarity measurement before running with `--similarity Euclidean`

# Example Commands

### Single Run
```
python main.py -d datax/ -m VAE-CF -i 200 -l 0.0000001 -r 100
```

### Hyper-parameter Tuning and Paper Result Reproduction

Split data in experiment setting, and tune hyper parameters based on yaml files in `config` folder. 
Please note the `reproduce_paper_results.py` will load pretrained model for Movielens1m from `latent` folder if that folder is not given in parameter.
```
python getmovielens.py --implicit -r 0.5,0.2,0.3 -d datax/ -n ml-1m/ratings.csv
python tune_parameters.py -d datax/ -n movielens1m/autorec.csv -y config/autorec.yml -gpu
python tune_parameters.py -d datax/ -n movielens1m/bpr.csv -y config/bpr.yml -gpu
python tune_parameters.py -d datax/ -n movielens1m/cdae.csv -y config/cdae.yml -gpu
python tune_parameters.py -d datax/ -n movielens1m/cml.csv -y config/cml.yml -gpu
python tune_parameters.py -d datax/ -n movielens1m/vae.csv -y config/vae.yml -gpu
python tune_parameters.py -d datax/ -n movielens1m/wrmf.csv -y config/wrmf.yml -gpu
python tune_parameters.py -d datax/ -n movielens1m/puresvd.csv -y config/puresvd.yml -gpu
python tune_parameters.py -d datax/ -n movielens1m/nceplrec.csv -y config/nceplrec.yml -gpu
python tune_parameters.py -d datax/ -n movielens1m/plrec.csv -y config/plrec.yml -gpu
```

Please check out the `cluster_bash` folder for further command details

Resplit data into two datasets: one for train, one for test. Note the train dataset includes validation set in previous split
```
python getmovielens.py --implicit -r 0.7,0.3,0.0 -d datax/ -n ml-1m/ratings.csv
python reproduce_paper_results.py -p tables/movielens1m -d datax/ -v Rvalid.npz -n movielens1m_test_result.csv -gpu
python reproduce_paper_results.py
```


