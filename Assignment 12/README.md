# DavidNet #

The objective is to reach 94% accuracy on Cifar10, which is reportedly human-level performance. Getting >94% Accuracy on Cifar10 means you can boast about building a super-human AI.
David C. Page is the winner of April - 2019 online competition DAWNBench to beat 94% accuracy. He has built a custom 9-layer Residual ConvNet, or ResNet. On a Tesla V100 GPU (125 tflops), DavidNet reaches 
94% with 75 seconds of training (excluding evaluation time). On Colab’s TPUv2 (180 tflops), we expect at least comparable performance — within 2 minutes as the TPU takes a long time to 
initialize. David C. Page has following method to reach this accuracy.

1. Preparing Data: Set up Google Cloud Storage (GCS) to store the entire data in memory. First download the dataset using Keras and do standard scaling: subtract by mean, and divide by standard deviation for every color channel.
2. In Tensorflow, the preferred file format is TFRecord, which is compact and efficient since it is based on Google’s ubiquitous ProtoBuf serialization library. Fenwicks provides a one-liner for this:        
``` python
fw.io.numpy_tfrecord(X_train, y_train, train_fn)    
fw.io.numpy_tfrecord(X_test, y_test, test_fn) 
```
3. Data augmentation and input pipeline: Training images go through the standard Cifar10 transformations, that is, pad 4 pixels to 40×40, crop back to 32×32, and randomly flip left and right. In addition, it applies the popular Cutout augmentation as a regularization measure, which alleviates overfitting. Cutout is a bit tricky to implement in Tensorflow. Fortunately, Fenwicks again provides one-liners:
``` python
def parser_train(tfexample):
 x, y = fw.io.tfexample_numpy_image_parser(tfexample, img_size,
 img_size)
 x = fw.transform.ramdom_pad_crop(x, 4)
 x = fw.transform.random_flip(x)
 x = fw.transform.cutout(x, 8, 8)
return x, y

parser_test = lambda x: fw.io.tfexample_numpy_image_parser(x, img_size, img_size)
```
4. With the input parsers ready, he has built a input pipeline, with Fenwicks’ one-liner:
``` python
train_input_func = lambda params: fw.io.tfrecord_ds(train_fn, parser_train, batch_size=params['batch_size'], training=True)
eval_input_func = lambda params: fw.io.tfrecord_ds(test_fn, parser_test, batch_size=params['batch_size'], training=False)
```
5. Building ConvNet. Building a ConvNet in Fenwicks isn’t hard — we just keep adding layers to a Sequential model as in the good-old Keras. For DavidNet, things are a bit tricky because the original implementation is in PyTorch. There are some subtle differences between PyTorch and Tensorflow. Most notably, PyTorch’s default way to set the initial, random weights of layers does not have a counterpart in Tensorflow. Fenwicks takes care of that. The ConvNet is as built as follows:

<img src="https://github.com/vinayakumarvs/EVA/blob/master/Assignment%2012/DavidNetArch.png" width="100%" height="50%">
</centre>

```python
def build_nn(c=64, weight=0.125):
 model = fw.Sequential()
 model.add(fw.layers.ConvBN(c, **fw.layers.PYTORCH_CONV_PARAMS))
 model.add(fw.layers.ConvResBlk(c*2, res_convs=2,
 **fw.layers.PYTORCH_CONV_PARAMS))
 model.add(fw.layers.ConvBlk(c*4, **fw.layers.PYTORCH_CONV_PARAMS))
 model.add(fw.layers.ConvResBlk(c*8, res_convs=2,
 **fw.layers.PYTORCH_CONV_PARAMS))
 model.add(tf.keras.layers.GlobalMaxPool2D())
 model.add(fw.layers.Classifier(n_classes, 
 kernel_initializer=fw.layers.init_pytorch, weight=0.125))
return model
```
6. Model training. DavidNet trains the model with Stochastic Gradient Descent with Nesterov momentum, with a slanted triangular learning rate schedule. Build the learning rate schedule and build the SGD optimizer and model function for TPUEstimator and trained the model. After the slowish initialization and first epoch, each epoch takes around 2.5 seconds. Since there are 24 epochs in total, the total amount of time spent on training is roughly a minute. By evaluating the model on the test set to see the accuracy was most of the time >94%.

## Reference: https://mc.ai/tutorial-2-94-accuracy-on-cifar10-in-2-minutes/ ##

In an earlier attempt the same network has been trained by David in 79 seconds on a single V100 GPU, comfortably beating the winning multi-GPU time, with plenty of room for improvement. The steps followed are:

1. Baseline: Analysed a baseline and remove a bottleneck in the data loading. (training time: 297s)
Established a baseline for training a Residual network to 94% test accuracy on CIFAR10, which takes 297s on a single V100 GPU. The network used in the fastest submission was an 18-layer Residual network, shown below. In this case the number of layers refers to the serial depth of (purple) convolutional and (blue) fully connected layers although the terminology is by no means universal:
<img src="https://github.com/vinayakumarvs/EVA/blob/master/Assignment%2012/Artboard-1-5.svg" width="100%" height="50%" >
</centre>
The network was trained for 35 epochs using SGD with momentum and the slightly odd learning rate schedule. David built a version of the network in PyTorch and replicated the learning rate schedule and hyperparameters from the DAWNBench submission. Training on an AWS p3.2×large instance with a single V100 GPU, 3/5 runs reach a final test accuracy of 94% in 356s. With this baseline improvement A first observation: the network starts with two consecutive (yellow-red) batch norm-ReLU groups after the first (purple) convolution. This was presumably not an intentional design and so let’s remove the duplication. Likewise the strange kink in the learning rate at epoch 15 has to go although this shouldn’t impact training time. With those changes in place the network and learning rate look slightly simpler and more importantly, 4/5 runs reach 94% final test accuracy with a new record in a time of 323s! 

A second observation: some of the image preprocessing (padding, normalisation and transposition) is needed on every pass through the training set and yet this work is being repeated each time. Other preprocessing steps (random cropping and flipping) differ between epochs and it makes sense to delay applying these. Although the preprocessing overhead is being mitigated by using multiple CPU processes to do the work, it turns out that PyTorch dataloaders (as of version 0.4) launch fresh processes for each iteration through the dataset. The setup time for this is non-trivial, especially on a small dataset like CIFAR10. By doing the common work once before training, removing pressure from the preprocessing jobs, we can reduce the number of processes needed to keep up with the GPU down to one. In heavier tasks, requiring more preprocessing or feeding more than one GPU, an alternative solution could be to keep dataloader processes alive between epochs. In any case, the effect of removing the repeat work and reducing the number of dataloader processes is a further 15s saving in training time (almost half a second per epoch!) and a new training time of 308s.

A bit more digging reveals that most of the remaining preprocessing time is spent calling out to random number generators to select data augmentations rather than in the augmentations themselves. During a full training run we make several million individual calls to random number generators and by combining these into a small number of bulk calls at the start of each epoch we can shave a further 7s of training time. Finally, at this point it turns out that the overhead of launching even a single process to perform the data augmentation outweighs the benefit and we can save a further 4s by doing the work on the main thread, leading to a final training time of 297s.

2. Mini-batches: Increased the size of mini-batches. Things go faster and don’t break. We investigate how this can be. (training time: 256s)
Further trained the model with batch size 512. Training completes in 256s and with one minor adjustment to the learning rate – increasing it by 10% – enabled to match the training curve of the base runs with batch size 128 and 3/5 runs reach 94% test accuracy. The noisier validation results during training at batch size 512 are expected because of batch norm effects. Larger batches may also be possible with a little care, but for now settled for 512.

In order to train a neural network at high learning rates then there are two regimes to consider. For the current model and dataset, at batch size 128 we are safely in the regime where forgetfulness dominates and we should either focus on methods to reduce this (e.g. using larger models with sparse updates or perhaps natural gradient descent), or we should push batch sizes higher. At batch size 512 we enter the regime where curvature effects dominate and the focus should shift to mitigating these.
3. Regularisation: We remove a speed bump in the code and add some regularisation. Our single GPU is faster than an eight GPU competition winner. (training time: 154s)
4. Architecture: We search for more efficient network architectures and find a 9 layer network that trains well. (training time: 79s)
5. Hyperparameters: We develop some heuristics to aid with hyperparameter tuning.
6. Weight decay: We investigate how weight decay controls the learning rate dynamics.
7. Batch norm: We learn that batch normalisation protects against covariate shift after all.
8. Bag of tricks: We uncover many ways to speed things up further when we find ourselves displaced from the top of the leaderboard. (final training time: 26s)
