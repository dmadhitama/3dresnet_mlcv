python main.py --root_path data/hvu/ --train_path frames_val --val_path frames_val --annotation_path anno_hvu.json --result_path results --dataset hvu --model resnet --model_depth 50 --n_classes 3142 --batch_size 32 --n_threads 16 --checkpoint 5 --sample_t_stride 2 --sample_duration 32 --no_val