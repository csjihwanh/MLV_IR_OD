echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:/scratch/e1640a06/MLV_IR_OD
export CUDA_VISIBLE_DEVICES='0,1'

echo "PYTHONPATH: ${PYTHONPATH}"

NNODE=1
NUM_GPUS=2
NUM_CPU=128

python task/test.py

