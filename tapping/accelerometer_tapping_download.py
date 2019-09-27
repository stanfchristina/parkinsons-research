#!/usr/bin/env python
# coding: utf-8

# In[12]:


import json
import synapseclient
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

#syn = synapseclient.login(email='sample@gmail.com', password='myPassword', rememberMe=True)
syn = synapseclient.login()


# In[3]:


queried_tapping_table = syn.tableQuery(("SELECT 'recordId', 'healthCode', 'accel_tapping.json.items', 'medTimepoint' FROM syn5511439 WHERE medTimepoint IN ('Immediately before Parkinson medication', 'Just after Parkinson medication (at your best)')").format("syn5511439"))
tapping_df = queried_tapping_table.asDataFrame()
tapping_df['index'] = tapping_df.index


# In[4]:


json_files = syn.downloadTableColumns(queried_tapping_table, "accel_tapping.json.items")
items = json_files.items()


# In[5]:


tapping_json_files_temp = pd.DataFrame({"accel_tapping.json.items": [i[0] + '.0' for i in items], "accel_tapping_json_file": [i[1] for i in items]})
tapping_json_files_temp


# In[6]:


tapping_df["accel_tapping.json.items"] = tapping_df["accel_tapping.json.items"].astype(str)
tapping_df_temp = pd.merge(tapping_df, tapping_json_files_temp, on="accel_tapping.json.items")
tapping_df_temp


# In[13]:


mean_linear_acc_x = []
mean_linear_acc_y = []
mean_linear_acc_z = []

range_linear_acc_x = []
range_linear_acc_y = []
range_linear_acc_z = []

skew_rfft_linear_acc_x = []
skew_rfft_linear_acc_y = []
skew_rfft_linear_acc_z = []

kurtosis_rfft_linear_acc_x = []
kurtosis_rfft_linear_acc_y = []
kurtosis_rfft_linear_acc_z = []

sum_std_linear_acc = []
sum_variances_linear_acc = []

# Helper function to derive distance from time
def CalculateRange(a, t):
    velocity = []
    distance = []
    velocity.append(0)
    distance.append(0)
    for i in (range(len(a) - 1)):
        velocity.append(abs(a[i])*(t[i + 1] - t[i]) + velocity[i])
        distance.append(velocity[i + 1]*(t[i + 1] - t[i]))
    return sum(distance)

for row in tapping_df_temp["accel_tapping_json_file"]: 
    with open(row) as json_data:
        data = json.load(json_data)
        t = []
        x = []
        y = []
        z = []
        for item in data:
            t.append(item.get("timestamp"))
            x.append(item.get("x"))
            y.append(item.get("y"))
            z.append(item.get("z"))
            
        # Make time values 0 based/indexed
        t_zero = t[0]
        t[:] = [n - t_zero for n in t]
        
        t = np.array(t)
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        
        mean_linear_acc_x.append(np.mean(abs(x)))
        mean_linear_acc_y.append(np.mean(abs(y)))
        mean_linear_acc_z.append(np.mean(abs(z)))
        
        range_linear_acc_x.append(CalculateRange(x, t))
        range_linear_acc_y.append(CalculateRange(y, t))
        range_linear_acc_z.append(CalculateRange(z, t))
        
        # Compute Fourier transforms
        ft_x = abs(np.fft.rfft(x))
        ft_y = abs(np.fft.rfft(y))
        ft_z = abs(np.fft.rfft(z))
        
        skew_rfft_linear_acc_x.append(skew(ft_x, bias=False))
        skew_rfft_linear_acc_y.append(skew(ft_y, bias=False))
        skew_rfft_linear_acc_z.append(skew(ft_z, bias=False))
        
        kurtosis_rfft_linear_acc_x.append(kurtosis(ft_x, bias=False))
        kurtosis_rfft_linear_acc_y.append(kurtosis(ft_y, bias=False))
        kurtosis_rfft_linear_acc_z.append(kurtosis(ft_z, bias=False))
        
        sum_variances_linear_acc.append(np.var(x) + np.var(y) + np.var(z))
        sum_std_linear_acc.append(np.std(x) + np.std(y) + np.std(z))

# create new column in dataframe from list of averaged values
tapping_df_temp["mean_linear_x_accel"] = mean_linear_acc_x
tapping_df_temp["mean_linear_y_accel"] = mean_linear_acc_y
tapping_df_temp["mean_linear_z_accel"] = mean_linear_acc_z
tapping_df_temp["range_linear_x_accel"] = range_linear_acc_x
tapping_df_temp["range_linear_y_accel"] = range_linear_acc_y
tapping_df_temp["range_linear_z_accel"] = range_linear_acc_z
tapping_df_temp["skew_rfft_linear_x_accel"] = skew_rfft_linear_acc_x
tapping_df_temp["skew_rfft_linear_y_accel"] = skew_rfft_linear_acc_y
tapping_df_temp["skew_rfft_linear_z_accel"] = skew_rfft_linear_acc_z
tapping_df_temp["kurtosis_rfft_linear_x_accel"] = kurtosis_rfft_linear_acc_x
tapping_df_temp["kurtosis_rfft_linear_y_accel"] = kurtosis_rfft_linear_acc_y
tapping_df_temp["kurtosis_rfft_linear_z_accel"] = kurtosis_rfft_linear_acc_z
tapping_df_temp["sum_variances_linear_accel"] = sum_variances_linear_acc
tapping_df_temp["sum_std_linear_accel"] = sum_std_linear_acc

# Remove unnecessary columns
tapping_final = tapping_df_temp.drop(["accel_tapping.json.items", "accel_tapping_json_file"], axis=1)


# In[23]:


tapping_final = tapping_df_temp.drop(["index", "mean_x_accel", "mean_y_accel", "mean_z_accel", "accel_tapping.json.items", "accel_tapping_json_file"], axis=1)


# In[24]:


tapping_final


# In[25]:


tapping_final.to_csv('accel_tapping_reduced.csv')

