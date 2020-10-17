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
    nohup python -u src/tasks/vqa_lol_emb.py \
    --train train --valid minival  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERT snap/pretrained/model \
    --data $data \
    --batchSize 32 --optim bert --lr 1e-5 --epochs 15 \
    --tqdm --output $output ${@:4} > outputs/$name.lol_emb.out &

