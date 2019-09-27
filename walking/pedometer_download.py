#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import synapseclient
import numpy as np
import pandas as pd

#syn = synapseclient.login(email='sample@gmail.com', password='myPassword', rememberMe=True)
syn = synapseclient.login()


# In[2]:


ped_outbound = syn.tableQuery(("SELECT 'recordId', 'healthCode', 'pedometer_walking_outbound.json.items', 'medTimepoint' FROM syn5511449 WHERE medTimepoint IN ('Immediately before Parkinson medication', 'Just after Parkinson medication (at your best)')").format("syn5511449"))
files_ped_outbound = syn.downloadTableColumns(ped_outbound, "pedometer_walking_outbound.json.items")
items_ped_outbound = files_ped_outbound.items()


# In[3]:


temp_ped_outbound = pd.DataFrame({"pedometer_walking_outbound.json.items": [i[0] + '.0' for i in items_ped_outbound], "file_paths": [i[1] for i in items_ped_outbound]})


# In[6]:


ped_outbound = ped_outbound.asDataFrame()
ped_outbound["pedometer_walking_outbound.json.items"] = ped_outbound["pedometer_walking_outbound.json.items"].astype(str)


# In[8]:


result = pd.merge(ped_outbound, temp_ped_outbound, on='pedometer_walking_outbound.json.items')
result


# In[11]:


total_distance = []
total_steps = []

for row in result["file_paths"]: 
    with open(row) as json_data:
        data = json.load(json_data)
        
        distance = []
        num_steps = []
        
        if data is not None:
            for item in data:
                distance.append(item.get("distance"))
                num_steps.append(item.get("numberOfSteps"))

        distance = np.array(distance)
        num_steps = np.array(num_steps)
        
        total_distance.append(np.sum(distance))
        total_steps.append(np.sum(num_steps))

result["total_distance"] = total_distance
result["total_steps"] = total_steps

result = result.drop(["pedometer_walking_outbound.json.items", "file_paths"], axis=1)


# In[12]:


result


# In[13]:


result.to_csv('pedometer_outbound_reduced.csv')

