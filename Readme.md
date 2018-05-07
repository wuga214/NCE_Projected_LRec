Projected LRec
===

# Data
Spotify RecSys 2018 Competition Dataset.

Data is not suit to submit on github, so please prepare it yourself. It should be numpy npy file directly 
dumped from csr sparse matrix. It should be easy.. 

# Command
```
$ python main.py --disable-item-item -i 4 -l 80 -r 200
```

# Run-time

The entire running time is around half hour.  Note: this does not count the prediction step, 
where we need to multiply RQ and Y
 
```

================================================================================
|                              Parameter Setting                               |
================================================================================

Rank: 200
Lambda: 80
Mode: User based
SVD Iteration: 4
================================================================================
|                                 Loading Data                                 |
================================================================================

Elapsed: 00:00:03
Train U-I Dimensions: (1010000, 2262292)
Valid U-I Dimensions: (1010000, 2262292)
================================================================================
|                                Randomized SVD                                |
================================================================================

Elapsed: 00:09:40
================================================================================
|                       Closed-Form Linear Optimization                        |
================================================================================

Elapsed: 00:07:33
<class 'scipy.sparse.csc.csc_matrix'>
(200, 2262292)
<class 'scipy.sparse.csr.csr_matrix'>
(1010000, 200)
================================================================================
|                               Save U-V Matrix                                |
================================================================================

Elapsed: 00:03:53

```