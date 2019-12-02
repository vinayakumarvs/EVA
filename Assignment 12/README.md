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

## 1. Baseline: Analysed a baseline and remove a bottleneck in the data loading. (training time: 297s)
Established a baseline for training a Residual network to 94% test accuracy on CIFAR10, which takes 297s on a single V100 GPU. The network used in the fastest submission was an 18-layer Residual network, shown below. In this case the number of layers refers to the serial depth of (purple) convolutional and (blue) fully connected layers although the terminology is by no means universal:
<img src="https://github.com/vinayakumarvs/EVA/blob/master/Assignment%2012/Artboard-1-5.svg" width="100%" height="50%" >
</centre>
The network was trained for 35 epochs using SGD with momentum and the slightly odd learning rate schedule. David built a version of the network in PyTorch and replicated the learning rate schedule and hyperparameters from the DAWNBench submission. Training on an AWS p3.2×large instance with a single V100 GPU, 3/5 runs reach a final test accuracy of 94% in 356s. With this baseline improvement A first observation: the network starts with two consecutive (yellow-red) batch norm-ReLU groups after the first (purple) convolution. This was presumably not an intentional design and so let’s remove the duplication. Likewise the strange kink in the learning rate at epoch 15 has to go although this shouldn’t impact training time. With those changes in place the network and learning rate look slightly simpler and more importantly, 4/5 runs reach 94% final test accuracy with a new record in a time of 323s! 

A second observation: some of the image preprocessing (padding, normalisation and transposition) is needed on every pass through the training set and yet this work is being repeated each time. Other preprocessing steps (random cropping and flipping) differ between epochs and it makes sense to delay applying these. Although the preprocessing overhead is being mitigated by using multiple CPU processes to do the work, it turns out that PyTorch dataloaders (as of version 0.4) launch fresh processes for each iteration through the dataset. The setup time for this is non-trivial, especially on a small dataset like CIFAR10. By doing the common work once before training, removing pressure from the preprocessing jobs, we can reduce the number of processes needed to keep up with the GPU down to one. In heavier tasks, requiring more preprocessing or feeding more than one GPU, an alternative solution could be to keep dataloader processes alive between epochs. In any case, the effect of removing the repeat work and reducing the number of dataloader processes is a further 15s saving in training time (almost half a second per epoch!) and a new training time of 308s.

A bit more digging reveals that most of the remaining preprocessing time is spent calling out to random number generators to select data augmentations rather than in the augmentations themselves. During a full training run we make several million individual calls to random number generators and by combining these into a small number of bulk calls at the start of each epoch we can shave a further 7s of training time. Finally, at this point it turns out that the overhead of launching even a single process to perform the data augmentation outweighs the benefit and we can save a further 4s by doing the work on the main thread, leading to a final training time of 297s.

## 2. Mini-batches: Increased the size of mini-batches. Things go faster and don’t break. We investigate how this can be. (training time: 256s)
Further trained the model with batch size 512. Training completes in 256s and with one minor adjustment to the learning rate – increasing it by 10% – enabled to match the training curve of the base runs with batch size 128 and 3/5 runs reach 94% test accuracy. The noisier validation results during training at batch size 512 are expected because of batch norm effects. Larger batches may also be possible with a little care, but for now settled for 512.

In order to train a neural network at high learning rates then there are two regimes to consider. For the current model and dataset, at batch size 128 we are safely in the regime where forgetfulness dominates and we should either focus on methods to reduce this (e.g. using larger models with sparse updates or perhaps natural gradient descent), or we should push batch sizes higher. At batch size 512 we enter the regime where curvature effects dominate and the focus should shift to mitigating these.

## 3. Regularisation: We remove a speed bump in the code and add some regularisation. Our single GPU is faster than an eight GPU competition winner. (training time: 154s)

We can get a rough timing profile of our current setup by selectively removing parts of the computation and running the remainder. For example, we can preload random training data onto the GPU to remove data loading and transfer times. We can also remove the optimizer step and the ReLU and batch norm layers to leave just the convolutions. If we do this, we get the following rough breakdown of timings across a range of batch sizes:
<img src="https://github.com/vinayakumarvs/EVA/blob/master/Assignment%2012/timing_breakdown.svg" width="100%" height="50%" >
</centre>

A few things stand out. First, a large chunk of time is being spent on batch norm computations. Secondly, the main convolutional backbone (including pooling layers and pointwise additions) is taking significantly longer than the roughly one second predicted at 100% compute efficiency. Thirdly, the optimizer and dataloader steps don’t seem to be a major bottleneck and are not an immediate focus for optimization.

Quickly the team found the problem with batch norms – the default method of converting a model to half precision in PyTorch (as of version 0.4) triggers a slow code path which doesn’t use the optimized CuDNN routine. They have converted batch norm weights back to single precision then the fast code is triggered and things look much healthier:
<img src="https://github.com/vinayakumarvs/EVA/blob/master/Assignment%2012/timing_breakdown_b.svg" width="100%" height="50%" >
</centre>
With this improvement in place the time for a 35 epoch training run to 94% accuracy drops to 186s, closing in on target!

Cutting training to 30 epochs, would lead to a 161s finish, easily beating our current target, but simply accelerating the baseline learning rate schedule, leads to 0/5 training runs reaching 94% accuracy.

A simple regularisation scheme that has been shown to be effective on CIFAR10 is so-called Cutout regularisation which consists of zeroing out a random subset of each training image. We try this for random 8×8 square subsets of the training images, in addition to our standard data augmentation of padding, clipping and randomly flipping left-right.

Results on the baseline 35 epoch training schedule are promising with 5/5 runs reaching 94% accuracy and the median run reaching 94.3%, a small improvement over the baseline. A bit of manual optimization of the learning rate schedule (pushing the peak learning rate earlier and replacing the decay phase with a simple linear decay since the final epochs of overfitting don’t seem to help with the extra regularisation in place) brings the median run to 94.5%.

If we accelerate the learning rate schedule to 30 epochs, 4/5 runs reach 94% with a median of 94.13%. We can push the batch size higher to 768 and 4/5 reach 94% with a median of 94.06%. The timings for 30 epoch runs are 161s at batch size 512 and 154s at batch size 768, comfortably beating our target and setting what may be a new speed record for the task of training CIFAR10 to 94% test accuracy, all on a single GPU! For reference, the new 30 epoch learning rate schedule is plotted below. Other hyperparameters (momentum=0.9, weight decay=5e-4) are kept at their values from the original training setup.

<img src="https://github.com/vinayakumarvs/EVA/blob/master/Assignment%2012/new_learning_rate.svg" width="100%" height="50%" >
</centre>


## 4. Architecture: We search for more efficient network architectures and find a 9 layer network that trains well. (training time: 79s)

So far, David and team training a fixed network architecture, taken from the fastest single-GPU DAWNBench entry on CIFAR10. With some simple changes, they have reduced the time taken to reach 94% test accuracy from 341s to 154s. Further investigating to alternative architectures they could zero on the below architecture.

<img src="https://github.com/vinayakumarvs/EVA/blob/master/Assignment%2012/DavidNetArch.png" width="100%" height="50%">
</centre>

The rate of improvement from training for longer seems slow compared to the improvements achievable by using deeper architectures. This network achieves 93.8% test accuracy in 66s for a 20 epoch run. If we extend training to 24 epochs, 7 out of 10 runs reach 94% with a mean accuracy of 94.08% and training time of 79s!

We have found a 9 layer deep residual network which trains to 94% accuracy in 79s, cutting training time almost in half. One remaining question is did we really need the residual branches to reach 94% test accuracy? The answer to this is a clear no. For example the single branch network Extra:L1+L2+L3 reaches 95% accuracy in 180s with 60 epoch training and extra regularisation (12×12 cutout) and wider versions go higher still. But at least for now the fastest network to 94% is a residual network.
## 5. Hyperparameters: We develop some heuristics to aid with hyperparameter tuning.
## 6. Weight decay: We investigate how weight decay controls the learning rate dynamics.
## 7. Batch norm: We learn that batch normalisation protects against covariate shift after all.
## 8. Bag of tricks: We uncover many ways to speed things up further when we find ourselves displaced from the top of the leaderboard. (final training time: 26s)
