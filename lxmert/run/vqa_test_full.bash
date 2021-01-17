# The name of this experiment.
data=$2
model=$3
type=$4


# Save logs and models under snap/vqa; make backup.
output=data/vqa/$2

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python -u src/tasks/vqa_lol_mod.py \
    --test $type \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --load $model \
    --data $data \
    --batchSize 960 \
    --tqdm --output $output ${@:5}
