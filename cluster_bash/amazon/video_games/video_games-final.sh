#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/NCE_Projected_LRec
python reproduce_paper_results.py -p amazon/video_games -d data/amazon/video_games/ -n amazon_video_games_test_result.csv -gpu
