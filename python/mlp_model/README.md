# mlp_model
Code for training and inference of mlp models

1. Make sure data files are in the right place.
2. Run the script
```bash
bash run.sh
```

Notes

```main_feature_embedding.py``` and ```main_feature.py```

take input of training data and outputs checkpoint files. They have several arguments on data location, number of epochs, network architecture, etc.

```main_feature_embedding_lb.py``` and ```main_feature_lb.py```

take input of aforementioned checkpoints and do inference on validation, leaderboard or test set depending on the arguments

