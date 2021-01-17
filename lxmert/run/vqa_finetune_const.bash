# The name of this experiment.
name=$2
data=$3




# Save logs and models under snap/vqa; make backup.
output=snap/vqa/$name

if [ -d $output ] 
then
    echo "Directory $output exists."
    exit 2
fi

mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash



# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    nohup python -u src/tasks/vqa_const.py \
    --train train --valid minival,nominival  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --load /scratch/pbanerj6/VQA_LOL/models/snap/vqa/vqa_full_v2_exp5/BEST \
    --data $data \
    --batchSize 128 --optim bert --lr 5e-5 --epochs 30 \
    --tqdm --output $output ${@:4} > outputs/$name.const.out &

