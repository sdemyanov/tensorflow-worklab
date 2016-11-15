import os

DATA_DIR = '/home/sdemyanov/synapse/UniversalGAN/data/mnist'
RESULTS_DIR = '/home/sdemyanov/synapse/UniversalGAN/results/mnist'

#DATA_DIR = '/home/ge/Project/ICCV2017/lib/UniversalGAN/data/mnist'
#RESULTS_DIR = '/home/ge/Project/ICCV2017/lib/UniversalGAN/results/mnist'

TRAIN_FOLD = 'train'
VALID_FOLD = 'valid'
TEST_FOLD = 'test'

MODEL_NAME = 'model.ckpt'

RESTORING_FILE = None
PARAMS_FILE = os.path.join(RESULTS_DIR, 'params.json')

