# Data
root
├── navsim (containing the devkit)
├── exp
└── dataset
    ├── maps
    ├── navsim_logs
    |    ├── test
    |    ├── trainval
    └── sensor_blobs
         ├── test
         ├── trainval
    └── extra_data/planning_vb
         ├── trajectory_anchors_256.npy
         ├── formatted_pdm_score_256.npy
```
Set the required environment variables, by adding the following to your `~/.bashrc` file
Based on the structure above, the environment variables need to be defined as:
```
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
```

### Install the navsim-devkit
Finally, install navsim.
To this end, create a new environment and install the required dependencies:
```
conda env create --name navsim -f environment.yml
conda activate navsim
pip install -e .
```