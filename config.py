class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 204
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    train_root = 'datasets/hand/train'
    train_list = 'datasets/hand_img_train.txt'

    hand_feature_root='datasets/hand/test'
    hand_feature_file='datasets/hand_img_features.txt'
    feature_file='datasets/features.npy'

    lfw_root = 'datasets/hand/test'
    lfw_test_list = 'datasets/hand_img_test.txt'

    checkpoints_path = 'checkpoints'
    load_model_path = 'checkpoints/resnet18.pth'
    test_model_path = 'checkpoints/hand_resnet.pth'
    save_interval = 10

    train_batch_size = 16  # batch size
    test_batch_size = 60

    input_shape = (1, 128, 128)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 50
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
