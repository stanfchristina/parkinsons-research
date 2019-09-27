#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import pandas as pd
import synapseclient as sc

#syn = synapseclient.login(email='sample@gmail.com', password='myPassword', rememberMe=True)
syn = synapseclient.login()


# In[2]:


queried_tapping_table = syn.tableQuery(("SELECT 'recordId', 'healthCode', 'appVersion', 'tapping_results.json.TappingSamples', 'medTimepoint' FROM syn5511439 WHERE medTimepoint IN ('Immediately before Parkinson medication', 'Just after Parkinson medication (at your best)') AND NOT appVersion = 'version 1.0, build 7'").format("syn5511439"))
tapping_df = queried_tapping_table.asDataFrame()


# In[3]:


tapping_df


# In[4]:


json_files = syn.downloadTableColumns(queried_tapping_table, "tapping_results.json.TappingSamples")
file_paths = json_files.items()


# In[6]:


tapping_json_files_temp = pd.DataFrame({"tapping_results.json.TappingSamples": [i[0] + '.0' for i in file_paths], "file_path": [i[1] for i in file_paths]})
tapping_json_files_temp


# In[7]:


tapping_df["tapping_results.json.TappingSamples"] = tapping_df["tapping_results.json.TappingSamples"].astype(str)
merged = pd.merge(tapping_df, tapping_json_files_temp, on="tapping_results.json.TappingSamples")
merged


# In[9]:


none_button_count = []
left_button_count = []
right_button_count = []

total_tap_count = []
percent_correct = []

mean_tap_interval = []
min_tap_interval = []
max_tap_interval = []

for row in merged["file_path"]: 
    with open(row) as json_data:
        data = json.load(json_data)
        
        button_id = []
        time_stamps = []
        for item in data:
            button_id.append(item.get("TappedButtonId"))
            time_stamps.append(item.get("TapTimeStamp"))
            
        # Make time values 0 based/indexed
        t_zero = t[0]
        
        button_id = np.array(button_id)
        time_stamps = np.array(time_stamps)
        
        num_none = (button_id == "TappedButtonNone").sum()
        num_left = (button_id == "TappedButtonLeft").sum()
        num_right = (button_id == "TappedButtonRight").sum()
        num_all = len(button_id)
        
        none_button_count.append(num_none)
        left_button_count.append(num_left)
        right_button_count.append(num_right)
        
        total_tap_count.append(num_all)
        percent_correct.append((num_left + num_right) / num_all)
        
        # Need an array of all the time differences between the time stamps
        time_diffs = np.diff(time_stamps)
        mean_tap_interval.append(np.mean(time_diffs))
        min_tap_interval.append(np.min(time_diffs))
        max_tap_interval.append(np.max(time_diffs))


# In[8]:


t = [0, 1, 4, 5, 9, 12, 13]
v = np.diff(t)
v


# In[10]:


# create new column in dataframe from list of averaged values
merged["none_button_count"] = none_button_count
merged["left_button_count"] = left_button_count
merged["right_button_count"] = right_button_count
merged["total_tap_count"] = total_tap_count
merged["percent_correct"] = percent_correct
merged["mean_tap_interval"] = mean_tap_interval
merged["min_tap_interval"] = min_tap_interval
merged["max_tap_interval"] = max_tap_interval

# Remove unnecessary columns
tapping_final = merged.drop(["tapping_results.json.TappingSamples", "file_path", "appVersion"], axis=1)


# In[11]:


tapping_final


# In[12]:


tapping_final.to_csv('accuracy_tapping_reduced.csv')

