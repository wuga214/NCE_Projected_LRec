#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/NCE_Projected_LRec
python tune_parameters.py -d data/amazon/video_games/ -n amazon/video_games/wrmf-part8.csv -y config/wrmf-part8.yml
