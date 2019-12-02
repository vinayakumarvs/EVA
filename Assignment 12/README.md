# DavidNet #

The objective is to reach 94% accuracy on Cifar10, which is reportedly human-level performance. Getting >94% Accuracy on Cifar10 means you can boast about building a super-human AI.
David C. Page is the winner of April - 2019 online competition DAWNBench to beat 94% accuracy. He has built a custom 9-layer Residual ConvNet, or ResNet. On a Tesla V100 GPU (125 tflops), DavidNet reaches 
94% with 75 seconds of training (excluding evaluation time). On Colab’s TPUv2 (180 tflops), we expect at least comparable performance — within 2 minutes as the TPU takes a long time to 
initialize. David C. Page has following method to reach this accuracy.

1. Set up Google Cloud Storage (GCS) to store the entire data in memory.
2. Preparing Data: In Tensorflow, the preferred file format is TFRecord, which is compact and efficient since it is based on Google’s ubiquitous ProtoBuf serialization library. Fenwicks provides a one-liner for this:        
``` fw.io.numpy_tfrecord(X_train, y_train, train_fn) ```    
``` fw.io.numpy_tfrecord(X_test, y_test, test_fn) ```
3. 

## Reference: https://mc.ai/tutorial-2-94-accuracy-on-cifar10-in-2-minutes/ ##
