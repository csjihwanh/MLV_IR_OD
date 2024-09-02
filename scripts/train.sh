echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:/scratch/e1640a06/MLV_IR_OD
export CUDA_VISIBLE_DEVICES='0,1'

echo "PYTHONPATH: ${PYTHONPATH}"

NNODE=1
NUM_GPUS=2
NUM_CPU=3

python task/train.py \
    --epochs 10 \
    --batch 2 \
    --dataset "hscai"

