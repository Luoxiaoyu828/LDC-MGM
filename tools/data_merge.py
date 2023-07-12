#!/usr/bin/env python
# coding: utf-8

# In[21]:


import astropy.io.fits as fits
import os
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import pandas as pd


# In[ ]:





# In[10]:


item_str = '0145+010'
item_str1 = '0140+010'
data1 = r'/home/data/clumps_share/MWISP/R2_200/%s/%s_L.fits' % (item_str, item_str)
data2 = r'/home/data/clumps_share/MWISP/R2_200/%s/%s_L.fits' % (item_str1, item_str1)
mask_1 = r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/R2_200_detect/%s_L/LDC_auto_mask.fits' % item_str
mask_2 = r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/R2_200_detect/%s_L/LDC_auto_mask.fits' % item_str1


# In[135]:


def get_long_mask(pos_item, ii=1):
#     pos_item = ['%04d+015' % i for i in range(150, 124, -5)]
    mask_all = np.zeros([2411, 121, 421], np.int64)
    
    for i, pos_item_ in enumerate(pos_item):
        mask0 = r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/R2_200_detect/%s_L/LDC_auto_mask.fits' % pos_item_
        loc_outcat = r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/R2_200_detect/%s_L/LDC_auto_loc_outcat.csv' % pos_item_
        outcat_loc = pd.read_csv(loc_outcat, sep='\t')
        mask0_ = fits.getdata(mask0).astype(np.int64)
        mask0_sum = mask0_.sum(0)

        mask_loc = np.zeros(mask0_.shape, np.int64)

        for id_ in tqdm.tqdm(outcat_loc['ID'].values):

            mask_copy = mask0_.copy()
            x,y,v = np.where(mask_copy==int(id_))
            mask_loc[x,y,v] = ii
            ii += 1
        mask_all[:, :121, i*60 :i*60 + 121] += mask_loc
    return mask_all, mask_all.max()


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


# In[168]:


help(np.concatenate)


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


# In[150]:


pos_los_item


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


# In[59]:





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


# In[110]:


pos_item = ['%04d+015' % i for i in range(150, 124, -5)]
mask_all, id_num = get_long_mask(pos_item, ii=1)
plt.imshow(mask_all.sum(0))


# In[127]:


for i in tqdm.tqdm(range(0, 26, 5)):
    if (i - 15) < 0:
        syms = '-'
        pos_los_item.append('%s%03d' % (syms, abs(i - 15)))
    else:
        syms = '+'
        pos_los_item.append('%s%03d' % (syms, i - 15))

pos_los_item = ['-015', '-010', '-005', '+000', '+005', '+010']


# In[ ]:


iii = 1
mask_All = np.zeros([2411,421,421], np.int64) 
for i, lat_str in enumerate(pos_los_item[::-1]):
    pos_item = ['%04d%s' % (i, lat_str) for i in range(150, 124, -5)]
    print(pos_item)
    mask_all, id_num = get_long_mask(pos_item, ii=iii)
    iii = id_num
    mask_All[:,  i*60 :i*60 + 121, :] += mask_all


# In[132]:


mask_All[:,  i*60 :i*60 + 121, :].shape


# In[134]:


fig = plt.figure('MWISP', figsize=[10,10])
ax = fig.add_subplot(1,1,1)
ax.imshow(mask_All.sum(0))


# In[99]:


pos_item_
mask0 = r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/R2_200_detect/%s_L/LDC_auto_mask.fits' % pos_item_
mask1 = r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/R2_200_detect/%s_L/LDC_auto_mask.fits' % '0130+015'
mask0_ = fits.getdata(mask0).astype(np.int64)
mask0_sum = mask0_.sum(0)
mask1_ = fits.getdata(mask1).astype(np.int64)
mask1_sum = mask1_.sum(0)

fig = plt.figure('MWISP', figsize=[10,6])
ax = fig.add_subplot(1,2,1)
ax.imshow(mask1_sum)
ax1 = fig.add_subplot(1,2,2)
ax1.imshow(mask0_sum)


# In[ ]:





# In[111]:


id_num


# In[107]:



plt.plot(mask0_sum[:, 1])
plt.plot(mask1_sum[:, 61])


# In[86]:


fig = plt.figure('MWISP', figsize=[10,6])
ax = fig.add_subplot(1,2,1)
ax.imshow(mask_all.sum(0))
ax1 = fig.add_subplot(1,2,2)
ax1.imshow(mask_loc.sum(0))


# In[79]:


i*91 + 121


# In[28]:


pd.read_csv(r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/R2_200_detect/0105-005_L/LDC_auto_loc_outcat.csv')
os.path.exists(loc_outcat)
print('/home/data/clumps_share/data_luoxy/detect_R2_LDC/R2_200_detect/0105-005_L/LDC_auto_loc_outcat.csv')
print(loc_outcat)


# In[24]:


plt.imshow(data2_.sum(0))

