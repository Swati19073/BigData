#!/usr/bin/env python
# coding: utf-8

# In[6]:


#Reference: https://medium.com/@hannah15198/convert-csv-to-json-with-python-b8899c722f6d

import csv
import json
input_path='Group14.csv'
output_path='Group14.json'
data={}
count=1
with open(input_path) as file:
    csvReader=csv.DictReader(file)
    for rows in csvReader:
        ID=rows['ID']
        profiling_technique=rows['Profiling Technique']
        Dataset_ID=rows['Dataset ID']
        No_of_samples=rows['No of Samples']
        Type_of_samples=rows['Type of Samples']
        Pubmed_ID=rows['Pubmed ID']
        record=(ID,profiling_technique,Dataset_ID,No_of_samples,Type_of_samples,Pubmed_ID)
        data[count]=record
        count=count+1
print(data)
        
with open(output_path,'w') as file2:
    file2.write(json.dumps(data,indent=4))
        



# In[ ]:




