model_cfg = {
    'enc_num_layers': 4,
    'enc_channels': [64, 128, 256, 512],
    'dec_num_layers': 4,
    'dec_channels': [512, 256, 128, 64],
    'D_num_layers': 4,
    'D_channels': [64, 128, 256, 512]
}

train_cfg = {
    'work_dirs': './work_dirs/base',
    'train_set': './data/train',
    'val_set': './data/val',
    'pos_test_set': './data/pos_test',
    'neg_test_set': './data/neg_test',
    'batch_size': 64,
    'img_size': (64, 64),
    'epochs': 500,
    'd_lr': 2e-4,
    'g_lr': 5e-4,
    'd_optimizer': 'Adam',
    'g_optimizer': 'Adam',
    'alpha': 0.8 # weights for l1 loss in generator loss
}
