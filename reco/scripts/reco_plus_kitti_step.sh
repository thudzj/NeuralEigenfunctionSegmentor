#!/usr/bin/env bash

if [ "$#" -eq  "0" ]
   then
     python3 ../networks/reco_plus.py \
     --p_config "../configs/reco_plus_kitti_step.yaml"
 else
     python3 ../networks/reco_plus.py \
     --p_config "../configs/reco_plus_kitti_step.yaml" \
     --p_state_dict "$1"
fi
