#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from ..imports import *
# Mount Google Drive
from google.colab import drive # import drive from google colab

def Mount_Google_Drive (path = "/content/drive" ): # default location for the drive
#print(path)                 # print content of ROOT (Optional)
    drive.mount(path, force_remount=True) # we mount the google drive at /content/drive     

