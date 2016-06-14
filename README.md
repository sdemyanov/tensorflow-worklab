# tensorflow-worklab
Copyright (C) 2016 Sergey Demyanov

Email: my_name [at] my_sirname.net

This set of files is an example of how to use Tensorflow for importing and retraining an existing model, such as ResNet. The goal of this example is to separate all functionality on several classes, so it is easy to modify the parameters and run the models, while still having all code not too deep inside. 

CLASSES

- **Reader** - specifies the input, performs data augmentation and outputs training and testing batches. Input is specified as a list of image filenames and their lables. 

- **Network** - specifies the new model for training, name mapping to the pretrained model, loss function, learning rate coefficients.

- **Trainer** - contains the main training loop, and specifies the parameters for training (batch size, learning rates, number of steps, etc).

- **Tester** - contains the testing loop, and specifies the statistics to observe.

- **Session** - incorporates all initialization, running, saving and restoring operations.

- **Writer** - contains functions to write summaries for Tensorboard

The following scripts are created for launching:

- **train_and_test** - sequentially runs training and testing operations in order to track the performance

- **train** - runs only training for a specified number of iterations

- **test** - runs only testing for a specified number of iterations

- **test_many** - runs testing every time a new model appears in the saving folder. Can be run in parallel with training, but doubles the consumed amount of memory.

HOW TO RETRAIN A MODEL

In the Network class you specify your model. You start from defining the primitives (batch_norm, conv_layer, pool_layer, etc), based on that define the main network blocks (such as ResNet blocks), and define the whole network in the _construct function. No need to specify each block manually, you can use loops. For each and layer you can specify lr_mult, which is used to adjust the learning rate for this layer. If it is zero, the layer will remain fixed. 

Scopes are used to define the variable names, and visualize the graph in Tensorboard. Adjust them for your purpose. In order to find the variable names of an existing model, use the command

python /path/to/tensorflow/utils/inspect_checkpoint.py --file_name=/path/to/pretrained_model/model.ckpt

The inspect_checkpoint.py file can be found in 'tensorflow/python/tools' folder of the Tensorflow source, which you can download from GitHub.

This script will show you the variable names, their types and sizes. Use 'restscope' parameter to specify mapping of these filenames to your model, defined in Network class. If you want to adjust the architecture (for example, change the last layer on another one with a different number of classes), define the parameter 'restore=False' to specify that these variables need to be initialized from scratch and will not be restored.

Set up the path to the model to restore in Session class. For example, use [this link](https://raw.githubusercontent.com/ry/tensorflow-resnet/master/data/tensorflow-resnet-pretrained-20160509.tar.gz.torrent) to download pretrained ResNet models. A model to restore is used only at the start of training. Once the current session is saved (i.e. the checkpoint file exist), all variables are restored from it, including those with the parameter 'restore=False'. Therefore, you can stop and start training at any time.

HOW TO SPECIFY THE INPUT

Currently the reader file needs two list of image filenames and their labels, for training and testing. If your data is stored in other format, adjust the reader accoring to your needs using other examples of Tensorflow, such as MNIST and CIFAR-10.
