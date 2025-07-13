#!/bin/bash
#$ -M pliang@nd.edu
#$ -q gpu -l gpu=1
#$ -m abe
#$ -r y

data=Fluo-N2DH-SIM+
min_cc=30

export PATH=/afs/crc.nd.edu/user/p/pliang/.conda/envs/SparseConv/bin:$PATH
export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/p/pliang/.conda/envs/SparseConv/lib:$LD_LIBRARY_PATH


CODE_DIR=../Ceb
cd ${CODE_DIR}

python3 top_pose_classifier.py \
    --data_dir ./boundary_pm/${data}/train/ \
    --num_class 2 \
    --num_epoch 20 \
    --checkpoint_save_dir ./checkpoints/${data}/ \
    --model_name resnet \

python3 top_pose_classifier.py \
    --test_dir ./boundary_pm/${data}/test/ \
    --phase 'test' \
    --num_class 2 \
    --checkpoint ./checkpoints/${data}/epoch_019.pth \
    --result_dir ./result/${data} \
    --model_name resnet \


python3 get_results.py --img_dir ./final_ws/${data}/test/region/ \
  --pred_file ./result/${data}/negative_names.txt \
  --ws_line_dir ./final_ws/${data}/test/line/ \
  --ws_map_dir ./final_ws/${data}/test/map/ \
  --output_dir ./final_result/${data}/

