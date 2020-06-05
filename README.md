<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

**Team members**: 

Contact: 

## Environment

The model is implemented in Java and tested on the following environment:

* Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
* 256GB RAM
* Nvidia Titan V
* Java Oracle 1.8.0_171
* Apache Maven 3.3.9
* Intel MKL 2018.1.038
* XGBoost and XGBoost4j 1.0.0

## Execution

To parse and serialize your TSV data files run:
```
./parse.sh
```
Be sure to specify `RECSYS_DATA_PATH` to the directory where your TSV files are located (specifically `training.tsv`, `val.tsv`, `competition_test.tsv` must be present).

To train a model and produce a submission file run:
```
./run.sh
```
Be sure to specify `RECSYS_PATH` to the directory where your main RecSys 2020 is (this path will contain subdirectories of `Data/` and `Models/XGB/`).


## Results

#### Data Parsing

Output should look similar to below
```
RecSys20DataParser: getAllUsersAndTweets /data/competitions/recsys2020/Data/training.tsv [10000000] elapsed [2 min 20 sec] cur_spd [N/A]
RecSys20DataParser: getAllUsersAndTweets /data/competitions/recsys2020/Data/training.tsv [20000000] elapsed [3 min 22 sec] cur_spd [161314/s]
RecSys20DataParser: getAllUsersAndTweets /data/competitions/recsys2020/Data/training.tsv [30000000] elapsed [4 min 9 sec] cur_spd [184727/s]
RecSys20DataParser: getAllUsersAndTweets /data/competitions/recsys2020/Data/training.tsv [40000000] elapsed [4 min 59 sec] cur_spd [189644/s]
RecSys20DataParser: getAllUsersAndTweets /data/competitions/recsys2020/Data/training.tsv [50000000] elapsed [5 min 59 sec] cur_spd [182789/s]
RecSys20DataParser: getAllUsersAndTweets /data/competitions/recsys2020/Data/training.tsv [60000000] elapsed [6 min 58 sec] cur_spd [180417/s]
RecSys20DataParser: getAllUsersAndTweets /data/competitions/recsys2020/Data/training.tsv [70000000] elapsed [7 min 50 sec] cur_spd [181863/s]
RecSys20DataParser: getAllUsersAndTweets /data/competitions/recsys2020/Data/training.tsv [80000000] elapsed [8 min 36 sec] cur_spd [186507/s]
RecSys20DataParser: getAllUsersAndTweets /data/competitions/recsys2020/Data/training.tsv [90000000] elapsed [9 min 30 sec] cur_spd [186260/s]
RecSys20DataParser: getAllUsersAndTweets /data/competitions/recsys2020/Data/training.tsv [100000000] elapsed [10 min 15 sec] cur_spd [189647/s]
RecSys20DataParser: getAllUsersAndTweets /data/competitions/recsys2020/Data/training.tsv [110000000] elapsed [11 min 1 sec] cur_spd [192050/s]
RecSys20DataParser: getAllUsersAndTweets /data/competitions/recsys2020/Data/training.tsv [120000000] elapsed [11 min 56 sec] cur_spd [191029/s]
RecSys20DataParser: getAllUsersAndTweets /data/competitions/recsys2020/Data/training.tsv [130000000] elapsed [12 min 42 sec] cur_spd [192994/s]
RecSys20DataParser: getAllUsersAndTweets /data/competitions/recsys2020/Data/training.tsv [140000000] elapsed [13 min 38 sec] cur_spd [191774/s]
RecSys20DataParser: getAllUsersAndTweets /data/competitions/recsys2020/Data/val.tsv [150000000] elapsed [14 min 28 sec] cur_spd [192539/s]
RecSys20DataParser: getAllUsersAndTweets /data/competitions/recsys2020/Data/val.tsv [160000000] elapsed [15 min 26 sec] cur_spd [190902/s]
RecSys20DataParser: getAllUsersAndTweets /data/competitions/recsys2020/Data/competition_test.tsv [170000000] elapsed [16 min 6 sec] cur_spd [193875/s]
RecSys20DataParser: getAllUsersAndTweets  [173494755] elapsed [16 min 30 sec] cur_spd [192429/s]
RecSys20DataParser: Unique users 34365200 elapsed [16 min 30 sec]
RecSys20DataParser: Unique tweets 89664863 elapsed [16 min 30 sec]
RecSys20DataParser: parse [5000000] elapsed [18 min 0 sec] cur_spd [-5322/s]
RecSys20DataParser: parse [10000000] elapsed [19 min 14 sec] cur_spd [0/s]
RecSys20DataParser: parse [15000000] elapsed [20 min 26 sec] cur_spd [4608/s]
RecSys20DataParser: parse [20000000] elapsed [21 min 35 sec] cur_spd [8658/s]
RecSys20DataParser: parse [25000000] elapsed [22 min 55 sec] cur_spd [12147/s]
RecSys20DataParser: parse [30000000] elapsed [23 min 57 sec] cur_spd [15431/s]
RecSys20DataParser: parse [35000000] elapsed [25 min 1 sec] cur_spd [18372/s]
RecSys20DataParser: parse [40000000] elapsed [26 min 13 sec] cur_spd [20941/s]
RecSys20DataParser: parse [45000000] elapsed [27 min 18 sec] cur_spd [23367/s]
RecSys20DataParser: parse [50000000] elapsed [28 min 19 sec] cur_spd [25667/s]
RecSys20DataParser: parse [55000000] elapsed [29 min 25 sec] cur_spd [27705/s]
RecSys20DataParser: parse [60000000] elapsed [30 min 27 sec] cur_spd [29653/s]
RecSys20DataParser: parse [65000000] elapsed [31 min 23 sec] cur_spd [31570/s]
RecSys20DataParser: parse [70000000] elapsed [32 min 24 sec] cur_spd [33266/s]
RecSys20DataParser: parse [75000000] elapsed [33 min 29 sec] cur_spd [34785/s]
RecSys20DataParser: parse [80000000] elapsed [34 min 18 sec] cur_spd [36512/s]
RecSys20DataParser: parse [85000000] elapsed [35 min 18 sec] cur_spd [37921/s]
RecSys20DataParser: parse [90000000] elapsed [36 min 18 sec] cur_spd [39272/s]
RecSys20DataParser: parse [95000000] elapsed [37 min 16 sec] cur_spd [40558/s]
RecSys20DataParser: parse [100000000] elapsed [38 min 11 sec] cur_spd [41853/s]
RecSys20DataParser: parse [105000000] elapsed [39 min 17 sec] cur_spd [42861/s]
RecSys20DataParser: parse [110000000] elapsed [40 min 20 sec] cur_spd [43862/s]
RecSys20DataParser: parse [115000000] elapsed [41 min 6 sec] cur_spd [45151/s]
RecSys20DataParser: parse [120000000] elapsed [42 min 9 sec] cur_spd [46057/s]
RecSys20DataParser: parse [125000000] elapsed [42 min 54 sec] cur_spd [47264/s]
RecSys20DataParser: parse [130000000] elapsed [44 min 1 sec] cur_spd [47987/s]
RecSys20DataParser: parse [135000000] elapsed [45 min 4 sec] cur_spd [48760/s]
RecSys20DataParser: parse [140000000] elapsed [45 min 50 sec] cur_spd [49825/s]
RecSys20DataParser: parse [145000000] elapsed [46 min 56 sec] cur_spd [50452/s]
RecSys20DataParser: text_tokens 297 elapsed [1 hr 22 min 35 sec]
RecSys20DataParser: present_domains 0 elapsed [1 hr 22 min 44 sec]
RecSys20DataParser: tweet_type 3 elapsed [1 hr 23 min 37 sec]
RecSys20DataParser: language 11 elapsed [1 hr 24 min 33 sec]
RecSys20DataParser: present_media 2 elapsed [1 hr 24 min 59 sec]
RecSys20DataParser: hashtags 0 elapsed [1 hr 25 min 18 sec]
RecSys20DataParser: present_links 0 elapsed [1 hr 25 min 27 sec]
RecSys20DataParser: parse [5000000] elapsed [1 hr 29 min 30 sec] cur_spd [-956/s]
RecSys20DataParser: parse [10000000] elapsed [1 hr 31 min 11 sec] cur_spd [0/s]
RecSys20DataParser: parse [5000000] elapsed [1 hr 33 min 40 sec] cur_spd [-913/s]
RecSys20DataParser: parse [10000000] elapsed [1 hr 35 min 6 sec] cur_spd [0/s]
```

#### Feature Extraction

Output should look similar to below

```
RecSys20Model: starting LIBSVM TRAIN elapsed [0 sec]
RecSys20Model: data loaded elapsed [7 min 59 sec]
RecSys20Split: total 148075238 elapsed [31 sec]
RecSys20Split: totalPos 75649149 elapsed [31 sec]
RecSys20Split: totalNeg 72426089 elapsed [31 sec]
RecSys20Split: trainPos 11036177 elapsed [31 sec]
RecSys20Split: trainNeg 11036177 elapsed [31 sec]
RecSys20Split: validPos 1876276 elapsed [31 sec]
RecSys20Split: validNeg 1876276 elapsed [31 sec]
RecSys20Split: split done CURRENT_NEG_INDEX=12912453 elapsed [34 sec]
RecSys20FeatExtractor: initCache done elapsed [2 min 44 sec]
RecSys20NeighborCF: initR [10000000] elapsed [1 sec] cur_spd [N/A]
RecSys20NeighborCF: initR [20000000] elapsed [1 sec] cur_spd [11682243/s]
RecSys20NeighborCF: initR [30000000] elapsed [2 sec] cur_spd [11500863/s]
RecSys20NeighborCF: userCreator nRows:34365200 nCols:34365200 nnz:56539936 elapsed [28 sec]
RecSys20Model: train [2000000] elapsed [17 min 12 sec] cur_spd [N/A]
RecSys20Model: train [4000000] elapsed [21 min 35 sec] cur_spd [7608/s]
RecSys20Model: train [6000000] elapsed [24 min 39 sec] cur_spd [8948/s]
RecSys20Model: train [8000000] elapsed [25 min 32 sec] cur_spd [11999/s]
RecSys20Model: train [10000000] elapsed [26 min 23 sec] cur_spd [14510/s]
RecSys20Model: train [12000000] elapsed [27 min 13 sec] cur_spd [16645/s]
RecSys20Model: train [14000000] elapsed [28 min 3 sec] cur_spd [18437/s]
RecSys20Model: train [16000000] elapsed [28 min 54 sec] cur_spd [19953/s]
RecSys20Model: train [18000000] elapsed [29 min 44 sec] cur_spd [21269/s]
RecSys20Model: train [20000000] elapsed [30 min 36 sec] cur_spd [22383/s]
RecSys20Model: train [22000000] elapsed [31 min 27 sec] cur_spd [23383/s]
RecSys20Model: train [24000000] elapsed [32 min 19 sec] cur_spd [24257/s]
RecSys20Model: train [25824906] elapsed [33 min 30 sec] cur_spd [24364/s]
```

#### Training an XGB model

Output should look similar to below
```
RecSys20Model: starting XGB TRAIN for LIKE with highL2=false elapsed [0 sec]
RecSys20Model: train rows 22072354 elapsed [1 min 3 sec]
RecSys20Model: valid rows 3752552 elapsed [1 min 3 sec]
RecSys20Model: xbg params {colsample_bytree=0.8, tree_method=hist, base_score=0.1, seed=5, eval_metric=aucpr, max_depth=15, use_buffer=0, booster=gbtree, objective=binary:logistic, lambda=1, eta=0.1, alpha=0, max_bin=256, subsample=1, verbosity=2, gamma=0, min_child_weight=20} elapsed [1 min 3 sec]
RecSys20Model: nRounds 200 elapsed [1 min 3 sec]
...
```

#### Submitting XGB scores

Output should look similar to below
```
RecSys20Model: starting SUBMIT elapsed [0 sec]
RecSys20Model: data loaded elapsed [7 min 54 sec]
RecSys20Split: total 148075238 elapsed [37 sec]
RecSys20Split: totalPos 75649149 elapsed [37 sec]
RecSys20Split: totalNeg 72426089 elapsed [37 sec]
RecSys20Split: trainPos 11036177 elapsed [37 sec]
RecSys20Split: trainNeg 11036177 elapsed [37 sec]
RecSys20Split: validPos 1876276 elapsed [37 sec]
RecSys20Split: validNeg 1876276 elapsed [37 sec]
RecSys20Split: split done CURRENT_NEG_INDEX=12912453 elapsed [39 sec]
RecSys20FeatExtractor: initCache done elapsed [2 min 28 sec]
RecSys20NeighborCF: initR [10000000] elapsed [0 sec] cur_spd [N/A]
RecSys20NeighborCF: initR [20000000] elapsed [1 sec] cur_spd [17064846/s]
RecSys20NeighborCF: initR [30000000] elapsed [2 sec] cur_spd [16077170/s]
RecSys20NeighborCF: userCreator nRows:34365200 nCols:34365200 nnz:67327535 elapsed [29 sec]
RecSys20Model: submitting for LIKE elapsed [11 min 32 sec]
RecSys20Model: submit [1000000] elapsed [14 min 13 sec] cur_spd [N/A]
RecSys20Model: submit [2000000] elapsed [14 min 46 sec] cur_spd [29852/s]
RecSys20Model: submit [3000000] elapsed [15 min 20 sec] cur_spd [29778/s]
RecSys20Model: submit [4000000] elapsed [15 min 53 sec] cur_spd [29779/s]
RecSys20Model: submit [5000000] elapsed [18 min 12 sec] cur_spd [16708/s]
RecSys20Model: submit [6000000] elapsed [18 min 45 sec] cur_spd [18355/s]
RecSys20Model: submit [7000000] elapsed [19 min 18 sec] cur_spd [19656/s]
RecSys20Model: submit [8000000] elapsed [19 min 51 sec] cur_spd [20698/s]
RecSys20Model: submit [9000000] elapsed [20 min 24 sec] cur_spd [21536/s]
RecSys20Model: submit [10000000] elapsed [22 min 32 sec] cur_spd [18041/s]
RecSys20Model: submit [11000000] elapsed [23 min 5 sec] cur_spd [18783/s]
RecSys20Model: submit [12000000] elapsed [23 min 38 sec] cur_spd [19452/s]
RecSys20Model: submit [12984679] elapsed [24 min 27 sec] cur_spd [19503/s]
RecSys20Model: submit xgb inference done LIKE elapsed [28 min 29 sec]
```

# RETWEET

| Model | # of features | Rounds | Runtime (hours) | PRAUC (valid) | RCE (valid) | MRR (LB) | RCE (LB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| XGB |  | 200 |  |  |  |  |  |
| DL |  | 3000 |  |  |  |  |  |

# REPLY

| Model | # of features | Rounds | Runtime (hours) | PRAUC (valid) | RCE (valid) | MRR (LB) | RCE (LB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| XGB |  | 200 |  |  |  |  |  |
| DL |  | 3000 |  |  |  |  |  |

# LIKE

| Model | # of features | Rounds | Runtime (hours) | PRAUC (valid) | RCE (valid) | MRR (LB) | RCE (LB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| XGB |  | 200 |  |  |  |  |  |
| DL |  | 3000 |  |  |  |  |  |

# COMMENT

| Model | # of features | Rounds | Runtime (hours) | PRAUC (valid) | RCE (valid) | MRR (LB) | RCE (LB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| XGB |  | 200 |  |  |  |  |  |
| DL |  | 3000 |  |  |  |  |  |
