#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/NCE_Projected_LRec
python reproduce_paper_results.py -p amazon/digital_music -d data/amazon/digital_music/ -n amazon_digital_music_test_result.csv -gpu
