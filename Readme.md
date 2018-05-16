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

Data Path: /media/wuga/Experiments/IMPLEMENTATION_Projected_LRec/data/
Train File Name: Rtrain.npz
Valid File Name: Rvalid.npz
Algorithm: PLRec
Mode: Item-based
Alpha: 50
Rank: 50
Lambda: 1.0
SVD Iteration: 1
================================================================================
|                                 Loading Data                                 |
================================================================================

Elapsed: 00:00:00
Train U-I Dimensions: (138494, 131263)
================================================================================
|                                Randomized SVD                                |
================================================================================

Elapsed: 00:00:10
================================================================================
|                       Closed-Form Linear Optimization                        |
================================================================================

Elapsed: 00:00:08
================================================================================
|                               Save U-V Matrix                                |
================================================================================

Elapsed: 00:00:00
================================================================================
|                                Create Metrics                                |
================================================================================

100%|██████████████████████████████████| 138494/138494 [15:18<00:00, 150.71it/s]

NDCG :0.172037321697
R-Precision :0.117117117117
Clicks :0.0
Elapsed: 00:15:19

```