import argparse
from pathlib import Path


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',
                        default=None,
                        type=Path,
                        help='Root directory path')
    
    parser.add_argument('--train_path',
                        default=None,
                        type=Path,
                        help='Directory path of train frames')
    
    parser.add_argument('--val_path',
                        default=None,
                        type=Path,
                        help='Directory path of validation frames')
    
    parser.add_argument('--annotation_path',
                        default=None,
                        type=Path,
                        help='Annotation file path')
    
    parser.add_argument('--result_path',
                        default=None,
                        type=Path,
                        help='Result directory path')
    
    parser.add_argument(
        '--dataset',
        default='kinetics',
        type=str,
        help='Used dataset (hvu)')
    
    parser.add_argument(
        '--n_classes',
        default=400,
        type=int,
        help=
        'Number of classes (hvu:3124)'
    )
    
    parser.add_argument('--sample_size',
                        default=112,
                        type=int,
                        help='Height and width of inputs')
    
    parser.add_argument(
        '--ft_begin_module',
        default='',
        type=str,
        help=('Module name of beginning of fine-tuning'
              '(conv1, layer1, fc, denseblock1, classifier, ...).'
              'The default means all layers are fine-tuned.'))
    
    parser.add_argument('--sample_duration',
                        default=16,
                        type=int,
                        help='Temporal duration of inputs')
    
    parser.add_argument(
        '--sample_t_stride',
        default=1,
        type=int,
        help='If larger than 1, input frames are subsampled with the stride.')
    
    parser.add_argument(
        '--train_crop',
        default='random',
        type=str,
        help=('Spatial cropping method in training. '
              'random is uniform. '
              'corner is selection from 4 corners and 1 center. '
              '(random | corner | center)'))
    
    parser.add_argument('--train_crop_min_scale',
                        default=0.25,
                        type=float,
                        help='Min scale for random cropping in training')
    
    parser.add_argument('--train_crop_min_ratio',
                        default=0.75,
                        type=float,
                        help='Min aspect ratio for random cropping in training')

    parser.add_argument('--train_t_crop',
                        default='random',
                        type=str,
                        help=('Temporal cropping method in training. '
                              'random is uniform. '
                              '(random | center)'))
    
    parser.add_argument('--learning_rate',
                        default=0.1,
                        type=float,
                        help=('Initial learning rate'
                              '(divided by 10 while training by lr scheduler)'))
    
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')

    parser.add_argument('--dampening',
                        default=0.0,
                        type=float,
                        help='dampening of SGD')
    
    parser.add_argument('--weight_decay',
                        default=1e-3,
                        type=float,
                        help='Weight Decay')
    
    parser.add_argument('--mean_dataset',
                        default='kinetics',
                        type=str,
                        help=('dataset for mean values of mean subtraction'
                              '(activitynet | kinetics | 0.5)'))
    
    parser.add_argument(
        '--value_scale',
        default=1,
        type=int,
        help=
        'If 1, range of inputs is [0-1]. If 255, range of inputs is [0-255].')
    
    parser.add_argument('--nesterov',
                        action='store_true',
                        help='Nesterov momentum')
    
    parser.add_argument('--optimizer',
                        default='sgd',
                        type=str,
                        help='Currently only support SGD')
    
    parser.add_argument('--lr_scheduler',
                        default='multistep',
                        type=str,
                        help='Type of LR scheduler (multistep | plateau)')
    
    parser.add_argument(
        '--multistep_milestones',
        default=[50, 100, 150],
        type=int,
        nargs='+',
        help='Milestones of LR scheduler. See documentation of MultistepLR.')

    parser.add_argument('--batch_size',
                        default=128,
                        type=int,
                        help='Batch Size')
    
    parser.add_argument('--n_epochs',
                        default=200,
                        type=int,
                        help='Number of total epochs to run')
    parser.add_argument('--n_val_samples',
                        default=3,
                        type=int,
                        help='Number of validation samples for each activity')
    
    parser.add_argument('--resume_path',
                        default=None,
                        type=Path,
                        help='Save data (.pth) of previous training')
    
    parser.add_argument('--no_train',
                        action='store_true',
                        help='If true, training is not performed.')
    
    parser.add_argument('--no_val',
                        action='store_true',
                        help='If true, validation is not performed.')
    
    parser.add_argument('--n_threads',
                        default=4,
                        type=int,
                        help='Number of threads for multi-thread loading')
    
    parser.add_argument('--checkpoint',
                        default=10,
                        type=int,
                        help='Trained model is saved at every this epochs.')
    
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help=
        '(resnet')
    
    parser.add_argument('--model_depth',
                        default=18,
                        type=int,
                        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    
    parser.add_argument('--conv1_t_size',
                        default=7,
                        type=int,
                        help='Kernel size in t dim of conv1.')
    
    parser.add_argument('--conv1_t_stride',
                        default=1,
                        type=int,
                        help='Stride in t dim of conv1.')
    
    parser.add_argument('--no_max_pool',
                        action='store_true',
                        help='If true, the max pooling after conv1 is removed.')
    
    parser.add_argument('--resnet_shortcut',
                        default='B',
                        type=str,
                        help='Shortcut type of resnet (A | B)')
    
    parser.add_argument(
        '--resnet_widen_factor',
        default=1.0,
        type=float,
        help='The number of feature maps of resnet is multiplied by this value')

    parser.add_argument('--input_type',
                        default='rgb',
                        type=str,
                        help='(rgb)')
    
    parser.add_argument('--manual_seed',
                        default=1,
                        type=int,
                        help='Manually set random seed')
    
    args = parser.parse_args()

    return args