Projected LRec
===

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

The entire running time is around half hour.  Note: this does not count the prediction step, 
where we need to multiply RQ and Y
 
```
================================================================================
|                              Parameter Setting                               |
================================================================================

Data Path: /media/wuga/Experiments/Recsys-18/IMPLEMENTATION_Projected_LRec/data/
movielens/
Data Name: raw/ratings.csv
Algorithm: WPLRec
Mode: Item-based
Alpha: 100
Rank: 100
Lambda: 10.0
SVD Iteration: 1
================================================================================
|                                 Loading Data                                 |
================================================================================

Elapsed: 00:01:47
Train U-I Dimensions: (138494, 131263)
================================================================================
|                                Randomized SVD                                |
================================================================================

Elapsed: 00:00:15
================================================================================
|                          Create Cacheable Matrices                           |
================================================================================

Elapsed: 00:00:02
================================================================================
|                           Item-wised Optimization                            |
================================================================================

  2%|▋                                  | 2602/131263 [01:50<1:26:10, 24.88it/s]
```