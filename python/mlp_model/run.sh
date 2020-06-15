#!/bin/bash


#run scaler
python scaler.py

python main_feature.py --run_name featurenet
python main_feature_lb.py -v Valid.sav -sp out/val/featurenet
python main_feature_lb.py -v Submit.sav -sp out/submit/featurenet
python main_feature_lb.py -v Test.sav -sp out/test/featurenet
