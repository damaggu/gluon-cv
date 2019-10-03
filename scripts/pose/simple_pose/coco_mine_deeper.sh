python train_simple_pose_mine.py \
    --model simple_pose_resnet152_v1d --mode hybrid --num-joints 14 --input-size 384,288 --sigma 3 \
    --lr 0.001 --wd 0.0 --lr-mode step --lr-decay-epoch 90,120 \
    --num-epochs 3000 --batch-size 8 --num-gpus 2 -j 60 \
    --dtype float32 --warmup-epochs 0 --use-pretrained-base \
    --save-dir params_simple_pose_resnet152_v1d \
    --logging-file simple_pose_resnet152_v1d_large_coco.log --log-interval 100
