Noise-contrastive Estimation Projected LRec
====================================================
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
![](https://img.shields.io/badge/mxnet-1.1.0-blue.svg)
![](https://img.shields.io/badge/matplotlib-3.0.0-blue.svg)

Please Disable GPU usage in `main.py` if needed.

# About NCE-PLRec
In multi-stage recommendation frameworks(see our [RecSys Challenge repository](https://github.com/layer6ai-labs/vl6_recsys2018)), the first stage requires scalable algorithms that can process millions users and items in reasonable time and still maintain good performance(recall 90%+). Even over almost 10 years, WRMF is still the most reliable algorithm for such task(see [RecSys Leaderboard](https://recsys-challenge.spotify.com/static/final_main_leaderboard.html)). Despite its good performance, the computation is too expensive and slow. Thus, we introduce a fast linear recommendation algorithm called NCE-PLRec to replace WRMF in large scale recommendation tasks. More details about this work are well described in our paper.

If you are interested in building up your research on this work, please cite:
```
@ARTICLE {wuga18nceplrec,
    author  = "Ga Wu, Maksims Volkovs, Chee Loong Soon, Scott Sanner, Himanshu Rai",
    title   = "Noise Contrastive Estimation for Scalable Linear Models for One-Class Collaborative Filtering",
    journal = "arXiv: 1811.00697",
    year    = "2018"
}
```


# Algorithm Implemented
1. Noise-contrastive Estimation Projected LRec(NCE-PLRec)
2. Linear FLow or Projected LRec(PLRec)
3. Collaborative Metric Learning(CML)
4. Auto-encoder Recommender(AutoRec)
5. Collaborative Denoising Auto-Encoders(CDAE)
6. Weighted Regularized Matrix Factorization(WRMF)
7. Pure SVD Recommender(PureSVD)
8. Bayesian Personalized Ranking(BPR)
9. Item-Item Similarity*
10. Popularity

# Data
1. Movielens 1M,
2. Movielens 20M,
3. Yahoo 1R,
4. Netflix,
5. Amazon Prize,
6. and
7. Spotify RecSys 2018 Competition Dataset.

Data is not suit to submit on github, so please prepare it yourself. It should be numpy npy file directly 
dumped from csr sparse matrix. It should be easy.. 

# Command for Movielens 1M
Process data
```
python getmovielens.py --implicit -r 0.5,0.2,0.3 -d /NCE_Projected_LRec/datax/ \
-n ml-1m/ratings.csv --shape 6041 3953

```

Run Weighted NCE-PLRec
```
$ python main.py -i 4 -l 1.0 -r 100 -a 10 -m NCE-PLRec -d datax/ \
-t Rtrain.npz -v Rvalid.npz -k 10
```

# Measure
The above algorithm could be splitted into two major category based on the distance
measurement: Euclidean or Cosine. CML is a euclidean distance recommender. And, ALS 
is a typical Cosine distance recommender. When doing evaluation, please select 
similarity measurement before running with `--similarity Euclidean` 

# Run-time
The proposed NCE-PLRec model needs only 3 mins to process MovieLens 20M!

# Example Result
Note: reproduce this result is easy, but you need to tune the hyperparameter alpha to
see the improvement, since for even same dataset, the split of the dataset could cause
huge difference of the hyper-parameter requirement. We suggest to use 50% of the training
data to train and validation on the others. We also suggest to apply small alpha in order
to avoid poor generalization issue.

 
```
================================================================================
|                              Parameter Setting                               |
================================================================================

Data Path: datax/
Train File Name: Rtrain.npz
Valid File Name: Rvalid.npz
Algorithm: PLRec
Mode: Item-based
Alpha: 10
Rank: 100
Lambda: 1.0
SVD Iteration: 4
Evaluation Ranking Topk: 10
================================================================================
|                                 Loading Data                                 |
================================================================================

Elapsed: 00:00:00
Train U-I Dimensions: (6041, 3953)
================================================================================
|                                Randomized SVD                                |
================================================================================

Elapsed: 00:00:01
================================================================================
|                          Create Cacheable Matrices                           |
================================================================================

Elapsed: 00:00:00
================================================================================
|                           Item-wised Optimization                            |
================================================================================

100%|██████████████████████████████████████| 3953/3953 [00:17<00:00, 223.06it/s]
Elapsed: 00:00:17
================================================================================
|                               Save U-V Matrix                                |
================================================================================

Elapsed: 00:00:00
================================================================================
|                                Create Metrics                                |
================================================================================

100%|██████████████████████████████████████| 6041/6041 [00:08<00:00, 694.76it/s]
-
NDCG :0.182108679494
R-Precision :0.078431372549
Clicks :0.0
Elapsed: 00:00:08

```

# Popularity Analysis
![NDCG](figures/Density_hist_pop.png) <!-- .element height="50%" width="50%" -->

