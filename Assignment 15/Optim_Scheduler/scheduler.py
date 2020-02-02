#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class OneCycleLR:
    def __init__(self, epochs, learning_rate, batch_size, train_len):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.batches_per_epoch = train_len//batch_size + 1
        self.current_learning_rate = 0.0

    def lr_schedule(self, t):
        self.current_learning_rate = np.interp ([t], [0, (self.epochs + 1) // 5, self.epochs], [0, self.learning_rate, 0])[0]
        return self.current_learning_rate

    def get_lr_fun(self, global_step):
        return lambda : self.lr_schedule(global_step/self.batches_per_epoch)/self.batch_size

    def get_lr(self):
        return self.current_learning_rate

