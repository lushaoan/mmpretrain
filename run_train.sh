pip install einops
pip install mat4py
pip install rich

CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ./configs/barcode/1d/exp1.py 4 --work-dir ./work_dirs/1d/exp1/