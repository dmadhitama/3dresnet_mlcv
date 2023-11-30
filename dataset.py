from datasets.videodataset import VideoDataset
from datasets.loader import VideoLoader

def image_name_formatter(x):
    return f'img_{x:05d}.jpg'


def get_training_data(video_path,
                      annotation_path,
                      dataset_name,
                      spatial_transform=None,
                      temporal_transform=None,
                      target_transform=None):
    assert dataset_name in [
        'hvu'
    ]

    loader = VideoLoader(image_name_formatter)
    video_path_formatter = (lambda root_path, video_id: root_path / video_id)

    
    training_data = VideoDataset(video_path,
                                    annotation_path,
                                    'training',
                                    spatial_transform=spatial_transform,
                                    temporal_transform=temporal_transform,
                                    target_transform=target_transform,
                                    video_loader=loader,
                                    video_path_formatter=video_path_formatter)

    return training_data


def get_validation_data(video_path,
                        annotation_path,
                        dataset_name,
                        spatial_transform=None,
                        temporal_transform=None,
                        target_transform=None):
    assert dataset_name in [
        'hvu'
    ]

    loader = VideoLoader(image_name_formatter)
    video_path_formatter = (lambda root_path, video_id: root_path / video_id)
    
    validation_data = VideoDataset(video_path,
                                    annotation_path,
                                    'validation',
                                    spatial_transform=spatial_transform,
                                    temporal_transform=temporal_transform,
                                    target_transform=target_transform,
                                    video_loader=loader,
                                    video_path_formatter=video_path_formatter)

    return validation_data