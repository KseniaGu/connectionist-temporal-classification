import torch
from easydict import EasyDict

cfg = EasyDict()
cfg.paths = EasyDict()
cfg.paths.dataset = 'D:\\Datasets\\mjsynth\\mnt\\ramdisk\\max\\90kDICT32px'
cfg.paths.corrupted_images_paths = 'D:\\Datasets\\corrupted_images_paths.pickle'

cfg.images = EasyDict()
# small subset analysis data:
cfg.images.pix_mean = (0.4639679668548339, 0.4639679668548339, 0.4639679668548339)
cfg.images.pix_std = (0.26743203316124775, 0.26743203316124775, 0.26743203316124775)
# cfg.images.heights_min_max = (23, 32)
# cfg.images.widths_min_max = (30, 388)

cfg.train = EasyDict()
cfg.train.train_size = 10000
cfg.train.test_size = 891927
cfg.train.val_size = 802734
cfg.train.batch_size = 16
cfg.train.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cfg.train.init_lr = 0.0003
cfg.train.patience = 4
cfg.train.nrof_epochs = 350
cfg.train.ckpt_path = 'checkpoints/checkpoint'
cfg.train.logs_dir = 'logs'
cfg.train.log_interval = 100
cfg.train.load = False
