#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


rest = pd.read_csv("deviceMotion_rest_reduced.csv")
outbound = pd.read_csv("deviceMotion_outbound_reduced.csv")
combined = pd.merge(outbound, rest, how='inner', on=['recordId', 'healthCode'])
combined


# In[12]:


combined.dropna(inplace=True)
combined.reset_index(inplace=True)
combined.rename(columns={'medTimepoint_x': 'medTimepoint'}, inplace=True)
combined.drop(['Unnamed: 0_x', 'index_x', 'Unnamed: 0_y', 'index_y', 'medTimepoint_y'], axis=1, inplace=True)
combined.to_csv('deviceMotion_restAndOutbound.csv')


# In[13]:


combined

