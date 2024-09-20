echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:MLV_IR_OD
export CUDA_VISIBLE_DEVICES='0,1,2,3'

echo "PYTHONPATH: ${PYTHONPATH}"

python task/test.py \
    --dataset "hscai" \
    --checkpoint "/workspace/MLV_IR_OD/runs/detect/train48/weights/last.pt" \
    --device "3" \
    --save_dir "result_uni_trans_last_fin.txt" \
    --conf_threshold 0.01

