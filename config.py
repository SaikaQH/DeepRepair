
class DatasetConfig:
    # dataset name
    dataset = 'cifar10'

    # class number
    total_class_num = 10
    total_cluster_num = 5

    # original cifar 10 (extract into jpg format)
    # image: 01234_5.jpg: 01234->name, 5->label
    cifar10_path = "/EX_STORE/dRepair/CIFAR_10_img"

    # corruption sets of cifar 10 (npy format)
    cifar10c_path = "/EX_STORE/dRepair/CIFAR-10-C"

    # corruption list
    CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression'
    ]


class FailureCaseConfig:
    # failure image directory
    fail_img_dir = "./failure_case"

    # model type (eg. wrn, densenet, allconv)
    model_type = 'wrn'

    # corruption type
    corrup = 'gaussian_noise'

    # number of failure training data sampled from every class
    sample_num_per_class = 100


class StyleTransferConfig:
    
    # dataset name (eg. test, cifar10, tiny-imagenet)     # not support tiny-imagenet yet
    dataset = 'test'
    # dataset = DatasetConfig.dataset

    # source image dir
    content_dir = "./WCT2/examples/content"
    content_segment_dir = None
    # style image dir
    style_dir = "./WCT2/examples/style"
    style_segment_dir = None
    # output dir
    output_dir = "./WCT2/examples/outputs"

    # sample type
    sample_type = 'random'

    # select device
    gpu = 0

    # wct2 model's path
    wct_model_path = "/EX_STORE/dRepair_from_s4/dRepair_cifar10/DeepRepair_standart_ver/WCT2/model_checkpoints"