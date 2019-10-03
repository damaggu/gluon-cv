python train_simple_pose_mine.py \
    --model alpha_pose_resnet101_v1b_coco --mode hybrid --num-joints 14 \
    --input-size 320,256 --sigma 3 \
    --lr 0.001 --wd 0.0 --lr-mode step --lr-decay-epoch 90,120 \
    --num-epochs 1000 --batch-size 8 --num-gpus 2 -j 60 \
    --dtype float32 --warmup-epochs 0 \
    --save-dir params_simple_pose_resnet50_v1b \
    --logging-file simple_pose_resnet50_v1b.log --log-interval 100
