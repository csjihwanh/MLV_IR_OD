echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:/scratch/e1640a06/MLV_IR_OD
export CUDA_VISIBLE_DEVICES='0,1,2,3'

echo "PYTHONPATH: ${PYTHONPATH}"

python task/train.py \
    --epochs 200 \
    --batch 40 \
    --device "0,1,2,3" \
    --dataset "unified_dataset" \
    --optimizer "auto" \
    --model "yolov10mlv"

