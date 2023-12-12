def get_mean_std(value_scale, dataset):
    assert dataset in ['activitynet', 'kinetics', '0.5', 'hvu']

    if dataset == 'activitynet':
        mean = [0.4477, 0.4209, 0.3906]
        std = [0.2767, 0.2695, 0.2714]
    elif dataset == 'hvu':
        mean = [0.4345, 0.4051, 0.3775]
        std = [0.2768, 0.2713, 0.2737]
    elif dataset == '0.5':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif dataset == 'hvu':
        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]

    mean = [x * value_scale for x in mean]
    std = [x * value_scale for x in std]

    return mean, std