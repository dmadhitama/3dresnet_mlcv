CUDA_VISIBLE_DEVICES=0,1 python main.py --root_path data/hvu/ --train_path rawframes_train --val_path rawframes_val --annotation_path anno_hvu.json --result_path results/r3d18_gpu01_256_thread_16_duration_16_stride_2 --dataset hvu --model resnet --model_depth 18 --n_classes 3142 --batch_size 256 --n_threads 24 --checkpoint 5 --sample_t_stride 2 --sample_duration 16 --resume_path results/r3d18_gpu01_256_thread_16_duration_16_stride_2/save_105.pth --mean_dataset kinetics --learning_rate 0.001 --ft_begin_module 105