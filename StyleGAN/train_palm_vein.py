# train_palm_vein.py
import os
os.system(
    "python train.py "
    "--outdir=./training-runs "
    "--data=./datasets/palm_vein_lmdb "
    "--gpus=1 "
    "--cfg=stylegan3-t "
    "--batch=32 "
    "--gamma=8.2 "
    "--mirror=1 "
    "--snap=10"
)