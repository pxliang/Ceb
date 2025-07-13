#!/bin/bash
#$ -M pliang@nd.edu
#$ -q gpu -l gpu=1
#$ -m abe
#$ -r y

data=Fluo-N2DH-SIM+
min_cc=30


export PATH=/afs/crc.nd.edu/user/p/pliang/.conda/envs/SparseConv/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/p/pliang/.conda/envs/SparseConv/lib:$LD_LIBRARY_PATH

CODE_DIR=../Ceb_temporal
cd ${CODE_DIR}

boundary_result=../Ceb/final_result/${data}/
PM_dir=../examples/${data}/PM/test
WS_dir=../Ceb/final_ws/${data}/test/region/
WS_line_dir=../Ceb/final_ws/${data}/test/line/
WS_map_dir=../Ceb/final_ws/${data}/test/map/


python3 get_boundary_scores.py \
    --test_dir ../Ceb/boundary_pm/${data}/test/ \
    --phase 'test' \
    --num_class 2 \
    --checkpoint ../Ceb/checkpoints/${data}/epoch_019.pth \
    --model_name resnet \
    --WS_map_dir ${WS_map_dir} \
    --WS_line_dir ${WS_line_dir} \
    --PM_dir ${PM_dir}    \
    --reliable_file ./reliable_file/${data}/reliable_file.txt


export PATH=/afs/crc.nd.edu/user/p/pliang/.conda/envs/HEMD_v2/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/p/pliang/.conda/envs/HEMD_v2/lib:$LD_LIBRARY_PATH

python3 matching_v3.py \
  --img_dir ${PM_dir} \
  --watershed_dir ${WS_dir} \
  --watershed_map_dir ${WS_map_dir} \
  --watershed_line_dir ${WS_line_dir} \
  --base_img ./data/reliable/${data}/ \
  --iter 2 \
  --min_region ${min_cc} \
  --start 0 \
  --get_base 1 \
  --reliable_file ./reliable_file/${data}/reliable_file.txt \
  --low_thres 0.1 \
  --high_thres 0.5

python3 root_pad.py \
  --img_dir ./data/reliable/${data}/iter1/ \
  --watershed_dir ${WS_dir} \
  --app_result_dir ./data/results/${data}/ \
  --PM_dir ${PM_dir} \
  --extra_dir ${boundary_result}

