CUDA_VISIBLE_DEVICES=0,1 python main.py --root_path data/hvu/ --train_path rawframes_train --val_path rawframes_val --annotation_path anno_hvu.json --result_path results/rxnt50_gpu01_256_thread_16_duration_32_stride_5_mean_kinetics --dataset hvu --model resnext --model_depth 50 --n_classes 3142 --batch_size 128 --n_threads 16 --checkpoint 5 --sample_t_stride 5 --sample_duration 32 --mean_dataset kinetics --learning_rate 0.1 --resnext_cardinality 32
