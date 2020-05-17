#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Canny Edge detection

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('mild.jpg')
#280,320
edges = cv2.Canny(img,100,220)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image')

plt.subplot(122),plt.imshow(edges,cmap='gray')
plt.title('Edge Image')

plt.show()


# In[ ]:





# In[2]:


img.shape


# In[3]:


from CorpusCallosum import CorpusCallosum


# In[4]:


y = CorpusCallosum(img)


# In[5]:


y.plot_corpus_callosum()


# In[6]:


area = y.corpus_callosum_area()


# In[7]:


area


# In[ ]:




