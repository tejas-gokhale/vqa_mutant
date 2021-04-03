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
    python src/tasks/gqa_wsup_bins_3d_patches_embs_yn.py \
    --train train_yesno3 --valid testdev_yesno3 \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERT  /data/data/lxmert_data/snap/pretrained/model \
    --batchSize 64 --optim bert --epochs 20 \
    --tqdm --output $output ${@:3}


# CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
#     python -m torch.distributed.launch --nproc_per_node=${#strarr[*]} --master_port=$MASTER_PORT src/tasks/gqa_wsup_bins_3d_patches_embs_yn.py \
#     --train train_yesno --valid testdev_yesno \
#     --llayers 9 --xlayers 5 --rlayers 5 \
#     --loadLXMERT  /data/data/lxmert_data/snap/pretrained/model \
#     --batchSize 32 --optim bert --epochs 20 \
#     --tqdm --output $output ${@:3}
    
    
    
    
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
#     python -u src/tasks/gqa_wsup_bins_3d_patches_embs_yn.py \
#     --test testdev \
#     --llayers 9 --xlayers 5 --rlayers 5 \
#     --load /data/data/lxmert_data/snap/gqa/gqa_wsup_1e5_bins_lg_3d_patches_v6_yn/BEST \
#     --batchSize 64 \
#     --tqdm --output /data/data/lxmert_data/snap/gqa/gqa_wsup_1e5_bins_lg_3d_patches_v6_yn 
    
    
    
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 PYTHONPATH=$PYTHONPATH:./src \
#     python -m torch.distributed.launch --nproc_per_node=6 --master_port=12314 src/tasks/gqa_wsup_bins_3d_patches_embs_yn.py \
#     --train others --valid testdev_yesno \
#     --llayers 9 --xlayers 5 --rlayers 5 \
#     --loadLXMERT  /data/data/lxmert_data/snap/gqa/gqa_wsup_1e5_bins_lg_3d_patches_v6/BEST \
#     --batchSize 32 --optim bert --epochs 20 \
#     --tqdm --output /data/data/lxmert_data/snap/gqa/gqa_wsup_1e5_bins_lg_3d_patches_v6_eval/
    
    
