#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/NCE_Projected_LRec
python tune_parameters.py -d data/amazon/digital_music/ -n amazon/digital_music/cml-part3.csv -y config/cml-part3.yml
