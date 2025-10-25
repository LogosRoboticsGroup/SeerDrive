SPLIT=test  # SPLIT=trainval

export PYTHONPATH="/Path_To_SeerDrive"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/Path_To_Dataset/openscene/maps"
export NAVSIM_EXP_ROOT="/Path_To_SeerDrive/exp"
export NAVSIM_DEVKIT_ROOT="/Path_To_SeerDrive"
export OPENSCENE_DATA_ROOT="/Path_To_Dataset/openscene"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
split=$SPLIT \
cache.cache_path='/Path_To_SeerDrive/exp/metric_cache' \
scene_filter.frame_interval=1
