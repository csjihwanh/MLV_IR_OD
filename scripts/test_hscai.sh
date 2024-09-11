echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:MLV_IR_OD
export CUDA_VISIBLE_DEVICES='0,1,2,3'

echo "PYTHONPATH: ${PYTHONPATH}"

python task/test.py \
    --dataset "hscai" \
    --checkpoint "checkpoints/yolo10x_hscaionly_256epoch/best.pt"
