#!/bin/bash

data=BBBC039
min_cc=30
exp=train

source activate Ceb


CODE_DIR=../Ceb
cd ${CODE_DIR}

PM_dir=../examples/${data}/PM/${exp}
gtpath=../examples/${data}/GT/${exp}

python3 find_global_max.py --prob_dir ${PM_dir} \
  --min_cc ${min_cc} \
  --ws_result_dir "./global_max/${data}/${exp}/"

 module load matlab
 matlab -nodisplay -nosplash -nojvm -r "pre_processing('${PM_dir}', './global_max/${data}/${exp}/', './processed_pm/${data}/${exp}/');exit"

python3 modify_watershed.py \
  --prob_processed_dir ./processed_pm/${data}/${exp}/ \
  --marker_dir ./global_max/${data}/${exp}/ \
  --region_dir ./final_ws/${data}/${exp}/region/ \
  --line_dir ./final_ws/${data}/${exp}/line/ \
  --map_dir ./final_ws/${data}/${exp}/map/ \
  --min_cc ${min_cc}

python3 GT_matching.py \
  --pm_dir ${PM_dir} \
  --ref_dir ${gtpath} \
  --watershed_dir ./final_ws/${data}/${exp}/ \
  --result_dir ./final_ws/${data}/${exp}/boundary_label/


exp=test
PM_dir_test=../examples/${data}/PM/${exp}
gtpath_test=../examples/${data}/GT/${exp}

python3 find_global_max.py --prob_dir ${PM_dir_test} \
  --min_cc ${min_cc} \
  --ws_result_dir "./global_max/${data}/${exp}/"

module load matlab
matlab -nodisplay -nosplash -nojvm -r "pre_processing('${PM_dir_test}', './global_max/${data}/${exp}/', './processed_pm/${data}/${exp}/');exit"

python3 modify_watershed.py \
  --prob_processed_dir ./processed_pm/${data}/${exp}/ \
  --marker_dir ./global_max/${data}/${exp}/ \
  --region_dir ./final_ws/${data}/${exp}/region/ \
  --line_dir ./final_ws/${data}/${exp}/line/ \
  --map_dir ./final_ws/${data}/${exp}/map/ \
  --min_cc ${min_cc}

python3 GT_matching.py \
  --pm_dir ${PM_dir_test} \
  --ref_dir ${gtpath_test} \
  --watershed_dir ./final_ws/${data}/${exp}/ \
  --result_dir ./final_ws/${data}/${exp}/boundary_label/

python3 prepare_boundary.py --prob_dir ${PM_dir},${PM_dir_test} \
  --ws_boundary_dir ./final_ws/${data}/train/line/,./final_ws/${data}/test/line/ \
  --final_dir ./boundary_pm/${data}/train/,./boundary_pm/${data}/test/ \
  --ws_region_dir ./final_ws/${data}/train/region/,./final_ws/${data}/test/region/ \
  --ws_map_dir ./final_ws/${data}/train/boundary_label/,./final_ws/${data}/test/boundary_label/ \
  --max_point 10
