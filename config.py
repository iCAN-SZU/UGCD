import os

# -----------------
# DATASET ROOTS
# -----------------
os.environ.setdefault('DATASET_DIR', '../datasets')
dataset_dir = os.environ.get('DATASET_DIR')
cifar_10_root = os.path.join(dataset_dir, 'cifar10')
cifar_100_root = os.path.join(dataset_dir, 'cifar100')
cub_root = os.path.join(dataset_dir, 'cub')
aircraft_root = os.path.join(dataset_dir, 'fgvc-aircraft-2013b')
car_root = os.path.join(dataset_dir, 'cars')
herbarium_dataroot = os.path.join(dataset_dir, 'herbarium_19')
imagenet_root = os.path.join(dataset_dir, 'ImageNet')

# OSR Split dir
osr_split_dir = 'data/ssb_splits'

# -----------------
# OTHER PATHS
# -----------------
exp_root = 'dev_outputs' # All logs and checkpoints will be saved here