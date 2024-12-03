pip install einops
pip install mat4py
pip install rich
pip install fast_ctc_decode
pip install editdistance

CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ./configs/barcode/oned_decoder/exp_1.py 4 --work-dir ./work_dirs/oned_decoder/exp_1/