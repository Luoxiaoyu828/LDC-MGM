#!/usr/bin/env python
# coding: utf-8

# In[9]:


import astropy.io.fits as fits
import os
import matplotlib.pyplot as plt
import numpy as np
import tqdm


# In[10]:


item_str = '0145+010'
item_str1 = '0140+010'
data1 = r'/home/data/clumps_share/MWISP/R2_200/%s/%s_L.fits' % (item_str, item_str)
data2 = r'/home/data/clumps_share/MWISP/R2_200/%s/%s_L.fits' % (item_str1, item_str1)
mask_1 = r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/R2_200_detect/%s_L/LDC_auto_mask.fits' % item_str
mask_2 = r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/R2_200_detect/%s_L/LDC_auto_mask.fits' % item_str1


# In[11]:


def get_long_mask(pos_item):
    data0 = r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/R2_200_detect/%s_L/LDC_auto_mask.fits' % pos_item[0]
    data0_ = fits.getdata(data0)
#     print(pos_item)
    data_0 = data0_[:, :, :91]
    for i, item_str in enumerate(pos_item):
        if i == 0:
            st = 0
            end_ = 91
            continue
        elif  i == len(pos_item) - 1:
            st = 31
            end_ = 121
        else:
            st = 31
            end_ = 91
        data1 = r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/R2_200_detect/%s_L/LDC_auto_mask.fits' % item_str
        data1_ = fits.getdata(data1)
        data1_sum = data1_
        data_0 = np.concatenate([data_0, data1_sum[:, :, st:end_]], axis=2)
#         data_0 = np.hstack([data_0, data1_sum[:, :, st:end_]])
    print(data_0.shape)
    return data_0


# In[12]:


def get_long(pos_item):
    data0 = r'/home/data/clumps_share/MWISP/R2_200/%s/%s_L.fits' % (pos_item[0], pos_item[0])
    data0_ = fits.getdata(data0)
#     print(pos_item)
    data_0 = data0_[:, :, :91]
    for i, item_str in enumerate(pos_item):
        if i == 0:
            st = 0
            end_ = 91
            continue
        elif  i == len(pos_item) - 1:
            st = 31
            end_ = 121
        else:
            st = 31
            end_ = 91
        data1 = r'/home/data/clumps_share/MWISP/R2_200/%s/%s_L.fits' % (item_str, item_str)
        data1_ = fits.getdata(data1)
        data1_sum = data1_
        data_0 = np.concatenate([data_0, data1_sum[:, :, st:end_]], axis=2)
#         data_0 = np.hstack([data_0, data1_sum[:, :, st:end_]])
    print(data_0.shape)
    return data_0


# In[13]:


pos_item = ['%04d+015' % i for i in range(150, 124, -5)]
data_015 = get_long_mask(pos_item)    


# In[14]:


print(data_015.shape)
fig = plt.figure(figsize=[15,6])
ax = fig.add_subplot(1,1,1)
ax.imshow(data_015.sum(0))


# In[17]:


data_0 = data_015[:, 31:, :]
pos_los_item = []
for i in tqdm.tqdm(range(0, 26, 5)):
    if (i - 15) < 0:
        syms = '-'
        pos_los_item.append('%s%03d' % (syms, abs(i - 15)))
    else:
        syms = '+'
        pos_los_item.append('%s%03d' % (syms, i - 15))
    
for i, lat_str in enumerate(pos_los_item[::-1]):
    pos_item = ['%04d%s' % (i, lat_str) for i in range(150, 124, -5)]
#     print(pos_item)
    if i == 0:
        continue
    elif  i == len(pos_los_item) - 1:
        st = 0
        end_ = 91
    else:
        st = 31
        end_ = 91
   
    
    data_lat = get_long_mask(pos_item) 
#     print(st, end_)
    data_temp = data_lat[:, st:end_,:]
#     print(data_temp.shape)
    data_0 = np.concatenate([data_temp, data_0], axis=1)
#     data_0 = np.vstack([data_temp, data_0])

            


# In[18]:


print(data_0.shape)
fig = plt.figure(figsize=[15,6])
ax = fig.add_subplot(1,1,1)
ax.imshow(data_0.sum(0))


# In[61]:


data1_ = fits.getdata(data1)
data2_ = fits.getdata(data2)
mask1_ = fits.getdata(mask_1)
mask2_ = fits.getdata(mask_2)
print(data1_.shape, data2_.shape)
data1_sum = data1_.sum(0)
data2_sum = data2_.sum(0)

mask1_sum = mask1_.sum(0)
mask2_sum = mask2_.sum(0)


# In[62]:


fig = plt.figure('MWISP', figsize=[10,6])
ax = fig.add_subplot(1,2,1)
ax.imshow(data1_sum)
ax1 = fig.add_subplot(1,2,2)
ax1.imshow(data2_sum)


# In[63]:


fig = plt.figure('MWISP', figsize=[10,6])
ax = fig.add_subplot(1,2,1)
ax.imshow(mask1_sum)
ax1 = fig.add_subplot(1,2,2)
ax1.imshow(mask2_sum)


# In[52]:


fig = plt.figure('MWISP', figsize=[10,6])
ax = fig.add_subplot(1,2,1)
ax.imshow(data1_sum[:, :91])
ax1 = fig.add_subplot(1,2,2)
ax1.imshow(data2_sum[:, 31:])


# In[53]:


plt.plot(data1_sum[:, 91])
plt.plot(data2_sum[:, 31])


# In[54]:


data1_sum[:, 91] - data2_sum[:, 31]


# In[64]:


data_merge = np.hstack([data1_sum[:, :91], data2_sum[:, 31:]])
plt.imshow(data_merge)


# In[65]:


data_merge = np.hstack([mask1_sum[:, :91], mask2_sum[:, 31:]])
plt.imshow(data_merge)


# In[ ]:




