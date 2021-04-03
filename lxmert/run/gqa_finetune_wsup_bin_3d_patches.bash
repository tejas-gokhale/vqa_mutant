# The name of this experiment.
name=$2

# Save logs and models under snap/gqa; make backup.
output=/data/data/lxmert_data/snap/gqa/$name
mkdir -p  $output/src
cp -r src/*  $output/src/
cp $0  $output/run.bash

export MASTER_PORT=$((12000 + RANDOM % 20000))

IFS=','
read -a strarr <<< "$1"
echo "Number of GPUs ${#strarr[*]}"

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python -m torch.distributed.launch --nproc_per_node=${#strarr[*]} --master_port=$MASTER_PORT src/tasks/gqa_wsup_bins_3d_patches.py \
    --train train,valid --valid testdev \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERT  /data/data/lxmert_data/snap/pretrained/model \
    --batchSize 48 --optim bert --epochs 20 \
    --tqdm --output $output ${@:3}
