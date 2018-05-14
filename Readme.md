Weighted Projected LRec(AAAI-18)
================================

# Data
Movielens 20M and
Spotify RecSys 2018 Competition Dataset.

Data is not suit to submit on github, so please prepare it yourself. It should be numpy npy file directly 
dumped from csr sparse matrix. It should be easy.. 

# Command for Movielens 20M
```
$ python main.py -i 4 -a 100 -l 10 -r 100 -m WPLRec
-d /media/user/Experiments/Recsys-18/IMPLEMENTATION_Projected_LRec/data/movielens/
-n raw/ratings.csv --shape 138494 131263
```

# Run-time
This model needs only 3 mins to process MovieLens 20M!

The entire running time is around half hour.  Note: this does not count the prediction step, 
where we need to multiply RQ and Y
 
```
================================================================================
|                              Parameter Setting                               |
================================================================================

Data Path: /media/wuga/Experiments/IMPLEMENTATION_Projected_LRec/data/movielens/
Data Name: rating.npz
Algorithm: WPLRec
Mode: User-based
Alpha: 100
Rank: 100
Lambda: 10.0
SVD Iteration: 4
================================================================================
|                                 Loading Data                                 |
================================================================================

Elapsed: 00:00:00
Train U-I Dimensions: (138494, 131263)
================================================================================
|                                Randomized SVD                                |
================================================================================

Elapsed: 00:00:29
================================================================================
|                          Create Cacheable Matrices                           |
================================================================================

Elapsed: 00:00:02
================================================================================
|                           Item-wised Optimization                            |
================================================================================

100%|██████████████████████████████████| 138494/138494 [04:13<00:00, 545.78it/s]
================================================================================
|                               Save U-V Matrix                                |
================================================================================

U_100_10.0_WPLRec Shape: (138494, 100)
V_100_10.0_WPLRec Shape: (131263, 100)
Elapsed: 00:00:00
```