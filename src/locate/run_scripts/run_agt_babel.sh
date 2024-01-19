#!/bin/bash

#bash run_scripts/run_agt_trainer.sh babel 247 4 1 10 4 1e-5 learned 1e-5 30 500 4 4 256 4 512 0 1500 0 0 512
# snippet level, joint position
#bash run_scripts/run_agt_trainer.sh babel 60 1 1 10 2 1e-5 learned 1e-5 30 10 4 4 256 4 512 0 1500 0 0 100 69
# frame level, AR feature
#                                                                                                        HDIM
#CUDA_VISIBLE_DEVICES=3 bash run_scripts/run_agt_trainer.sh babel 73 1 1 10 30 1e-5 learned 1e-5 30 10 4 4 256 4 512 0 1500 0 0 150 256
#CUDA_VISIBLE_DEVICES=2 bash run_scripts/run_agt_trainer.sh babel 20 1 1 100 30 1e-5 learned 1e-5 30 20 4 4 256 4 512 0 1500 0 0 100 256
# Fixed position emb
#CUDA_VISIBLE_DEVICES=5 bash run_scripts/run_agt_trainer.sh babel 20 1 1 100 2 1e-5 nerf 1e-5 30 100 4 4 256 4 512 0 1500 0 0 100 600
CUDA_VISIBLE_DEVICES=6 bash run_scripts/run_agt_trainer.sh babel 20 1 1 100 2 1e-5 learned 1e-5 30 100 4 4 256 4 512 0 1500 0 0 100 600
#CUDA_VISIBLE_DEVICES=6 bash run_scripts/run_agt_trainer.sh babel 20 1 1 100 2 1e-5 learned 1e-5 30 100 4 4 256 4 512 0 1500 0 0 100 768
