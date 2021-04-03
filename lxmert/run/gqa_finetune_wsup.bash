# The name of this experiment.
name=$2

# Save logs and models under snap/gqa; make backup.
output=/data/data/lxmert_data/snap/gqa/$name
mkdir -p  $output/src
cp -r src/*  $output/src/
cp $0  $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/gqa_wsup.py \
    --train train,valid --valid testdev \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERT  /data/data/lxmert_data/snap/pretrained/model \
    --batchSize 256 --optim bert --epochs 20 \
    --tqdm --output $output ${@:3}
