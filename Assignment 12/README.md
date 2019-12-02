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
