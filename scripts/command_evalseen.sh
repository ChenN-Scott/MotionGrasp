CUDA_VISIBLE_DEVICES=0 python test.py \
--camera "realsense" \
--log_dir "../logs/log_train" \
--frame_size 5 \
--sequence_size 7 \
--batch_size 2 \
--model_name "pointnet" \
--dataset_root "/data/GraspNet_1billion" \
--evaluate_root "/data/one_billion" \
--feature_dim 128 \
--num_point 20000 \
--split "test" \
--test_type "seen" \
--checkpoint_path_2 "tracker_epoch_05.tar" \
--gt_dir "../gt/log_train" \
--dump_dir "../preds/log_train" \
--rot_type '6d' \
--eval \
--error