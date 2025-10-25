# Define
export PYTHONPATH="/Path_To_SeerDrive"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/Path_To_OpenScene/maps"
export NAVSIM_EXP_ROOT="/Path_To_SeerDrive/exp"
export NAVSIM_DEVKIT_ROOT="/Path_To_SeerDrive"
export OPENSCENE_DATA_ROOT="/Path_To_OpenScene"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CONFIG_NAME=default

### training ###
python ./navsim/planning/script/run_training.py \
agent=SeerDrive_agent \
agent.config._target_=navsim.agents.SeerDrive.configs.${CONFIG_NAME}.SeerDriveConfig \
experiment_name=train \
scene_filter=navtrain \
dataloader.params.batch_size=16 \
trainer.params.max_epochs=32  \
split=trainval 
