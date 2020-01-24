
### Assignment 14: 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinayakumarvs/EVA/blob/master/Assignment%2014/Assignment_14.ipynb)




# Model Training Logs
```
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.FixedLenFeature is deprecated. Please use tf.io.FixedLenFeature instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.parse_single_example is deprecated. Please use tf.io.parse_single_example instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.random_crop is deprecated. Please use tf.image.random_crop instead.

WARNING:tensorflow:From <ipython-input-8-b8cfd37e1cdb>:59: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
WARNING:tensorflow:From <ipython-input-8-b8cfd37e1cdb>:97: DatasetV1.make_one_shot_iterator (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `for ... in dataset:` to iterate over a dataset. If using `tf.estimator`, return the `Dataset` object directly from your input function. As a last resort, you can use `tf.compat.v1.data.make_one_shot_iterator(dataset)`.

epoch: 1 lr: 0.08 train loss: 1.6806415795898437 train acc: 0.38914 val loss: 1.2243451812744142 val acc: 0.5594 time: 60.7623028755188

epoch: 2 lr: 0.16 train loss: 0.9846619055175782 train acc: 0.64682 val loss: 0.7667159561157226 val acc: 0.7303 time: 110.20129656791687

epoch: 3 lr: 0.24 train loss: 0.7697196243286133 train acc: 0.72906 val loss: 0.7951693267822265 val acc: 0.736 time: 158.92873811721802

epoch: 4 lr: 0.32 train loss: 0.6656028549194336 train acc: 0.7672 val loss: 0.7483978088378906 val acc: 0.7528 time: 207.91357922554016

epoch: 5 lr: 0.4 train loss: 0.6120578289794922 train acc: 0.78822 val loss: 0.557934376525879 val acc: 0.8155 time: 256.8893356323242

epoch: 6 lr: 0.37894736842105264 train loss: 0.5127377392578125 train acc: 0.82178 val loss: 0.5635189849853516 val acc: 0.8256 time: 306.66149854660034

epoch: 7 lr: 0.35789473684210527 train loss: 0.4194139375305176 train acc: 0.85286 val loss: 0.5258476593017578 val acc: 0.827 time: 355.96831035614014

epoch: 8 lr: 0.33684210526315794 train loss: 0.3680939593505859 train acc: 0.87126 val loss: 0.47006245727539064 val acc: 0.8449 time: 404.8794758319855

epoch: 9 lr: 0.31578947368421056 train loss: 0.3172576823425293 train acc: 0.8899 val loss: 0.39801264419555665 val acc: 0.8668 time: 453.6769814491272

epoch: 10 lr: 0.2947368421052632 train loss: 0.2861446319580078 train acc: 0.89998 val loss: 0.410256950378418 val acc: 0.8666 time: 502.5094406604767

epoch: 11 lr: 0.2736842105263158 train loss: 0.2527581120300293 train acc: 0.9115 val loss: 0.33473280487060547 val acc: 0.8864 time: 551.2708539962769

epoch: 12 lr: 0.25263157894736843 train loss: 0.2275924188232422 train acc: 0.9198 val loss: 0.3406046859741211 val acc: 0.888 time: 599.7198417186737

epoch: 13 lr: 0.23157894736842108 train loss: 0.20322978073120118 train acc: 0.92972 val loss: 0.3227912239074707 val acc: 0.8936 time: 648.9327340126038

epoch: 14 lr: 0.2105263157894737 train loss: 0.17755862800598143 train acc: 0.93836 val loss: 0.3295452865600586 val acc: 0.8905 time: 697.4011785984039

epoch: 15 lr: 0.18947368421052635 train loss: 0.15777068321228027 train acc: 0.94552 val loss: 0.3253570991516113 val acc: 0.8925 time: 745.7488441467285

epoch: 16 lr: 0.16842105263157897 train loss: 0.14029958671569825 train acc: 0.95046 val loss: 0.3154951026916504 val acc: 0.8984 time: 793.9014747142792

epoch: 17 lr: 0.1473684210526316 train loss: 0.12136688911437989 train acc: 0.95856 val loss: 0.3434042175292969 val acc: 0.8951 time: 842.0621914863586

epoch: 18 lr: 0.12631578947368421 train loss: 0.10614925220489502 train acc: 0.96416 val loss: 0.3138007652282715 val acc: 0.9016 time: 890.6160085201263

epoch: 19 lr: 0.10526315789473689 train loss: 0.0937892744064331 train acc: 0.96868 val loss: 0.29969066848754883 val acc: 0.9077 time: 939.5716695785522

epoch: 20 lr: 0.08421052631578951 train loss: 0.0827849578857422 train acc: 0.97326 val loss: 0.2969021675109863 val acc: 0.9075 time: 987.6375949382782

epoch: 21 lr: 0.06315789473684214 train loss: 0.06974636688232422 train acc: 0.97704 val loss: 0.3017078201293945 val acc: 0.9122 time: 1035.734454870224

epoch: 22 lr: 0.04210526315789476 train loss: 0.06167115928649902 train acc: 0.98012 val loss: 0.2917001365661621 val acc: 0.9141 time: 1083.654104232788

epoch: 23 lr: 0.02105263157894738 train loss: 0.05616674644470215 train acc: 0.98182 val loss: 0.28425884323120115 val acc: 0.917 time: 1131.7750017642975

epoch: 24 lr: 0.0 train loss: 0.05436177185058594 train acc: 0.98338 val loss: 0.27814249038696287 val acc: 0.9181 time: 1179.8392450809479
```

### Training Graphs:
<img src="https://github.com/vinayakumarvs/EVA/blob/master/Assignment%2014/Training_Image.png" width="100%" height="50%">

