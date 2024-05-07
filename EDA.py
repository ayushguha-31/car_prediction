#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
import sys
from six.moves import urllib
warnings.filterwarnings('ignore')

# if "ipykernel" in sys.modules:
#     get_ipython().run_line_magic('matplotlib', 'inline')


# # Download and import the csv data as pandas dataframe

# In[14]:


df=pd.read_csv('cardekho_dataset.csv')


# # Show Top 5 Records

# In[15]:


df.head(5)


# In[20]:


df.columns


# In[21]:


df.drop(columns='Unnamed: 0',inplace=True)


# 
# # Shape of Dataset

# In[22]:


df.shape


# 
# # Summary of Data
# 

# In[23]:


## display summary statistics for dataframe


# In[24]:


df.describe()


# # Check Datatypes in the Dataset

# In[25]:


df.info()


# # Exploring Data

# In[26]:


#define numerical $ categorical columns


# In[29]:


numeric_feat=[feat for feat in df.columns if df[feat].dtype!='O']
categorical_feat=[feat for feat in df.columns if df[feat].dtype=='O']

#print columns 
print('We have {} numerical features:{}'.format(len(numeric_feat),numeric_feat))
print('We have {} catergorical features:{}'.format(len(categorical_feat),categorical_feat))


# # # Feature Information
# car_name: Car's full name , which includes brand and specific model name
# brand:Brand name for a particular car
# model:Exact model name of the car of a particular brand name
# seller_type:Which type of seller is selling the used car
# fuel_type:fuel used in the car, which was put up on the sale
# transmission_type: Transmission used in the used car , which was put on the sale
# vehicle_age: The count of years since the car was bought
# mileage: It is the number of kms the car runs per litre
# engine: It is the engine capacity in cc
# max_power : Max power it produces in BHP
# seats : Total number of seats in the car
# selling_price : The sale price which was put on the website
# In[30]:


## proportion of count data on categorical columns
for col in categorical_feat:
    print(df[col].value_counts(normalize=True)*100)
    print('------------------------------------------')


# # Univariate Analysis 

# # Numerical Features

# In[32]:


plt.figure(figsize=(15,15))
plt.suptitle('univariate analysis of numerical features',fontsize=20)
for i in range(0,len(numeric_feat)):
    plt.subplot(3,3,i+1)
    sns.kdeplot(x=df[numeric_feat[i]],shade=True,color='b')
    plt.xlabel(numeric_feat[i])
    plt.tight_layout()

# Reports
# km_driven,engine,max_power,selling_price are right skewed and positively skewed
# outliers in km_driven , engine, selling_price, max_power
# # Categorical Features

# In[35]:


plt.figure(figsize=(20,15))
plt.suptitle('univariate analysis of categorical features',fontsize=20)
cat1=['brand','seller_type','fuel_type','transmission_type']
for i in range (0,len(cat1)):
    plt.subplot(2,2,i+1)
    sns.countplot(x=df[cat1[i]])
    plt.xlabel(cat1[i])
    plt.xticks(rotation=45)
    plt.tight_layout()

# Report
# Marutii has the most number of cars
# People find dealers more trustworthy than other two categories
# Petrol and diesel compete in the fuel category
# User still prefer manually transmitted cars 
# # Multivariate Analysis 

# In[37]:


df.corr(numeric_only=True)


# In[39]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(numeric_only=True),cmap='CMRmap',annot=True)
plt.show()

# Report
# mileage and engine are negatively correlated
# max power has highest positive correlation with target variable selling price
# km_driven has least correlation with target variable 
# # Checking Null values in the dataset

# In[40]:


df.isnull().sum()


# # Continuous Feature and Selling Price

# In[43]:


continous_features=[feat for feat in numeric_feat if len(df[feat].unique())>=30]
print('num of continous features:',continous_features)


# In[44]:


fig = plt.figure(figsize=(12,10))
for i in range(0,len(continous_features)):
    if continous_features[i]!='selling_price':
        ax=plt.subplot(2,2,i+1)
        sns.scatterplot(data=df,y='selling_price',x=continous_features[i],color='b')
        plt.tight_layout()

# Report 
# km driven has -ve relationship
# engine has +ve relationship
# mileage has low -ve correlation 
# max power has +ve correlation
# # # Intial Analysis Report
# Lower vehicle age has more selling price than vehicle with more age
# Engine has positive effect on the price
# Kms driven has negative effect on the price 
# No null values in the data set
# # # Visualization 

# In[45]:


plt.subplots(figsize=(14,7))
sns.histplot(df.selling_price, bins=200, kde=True, color = 'b')
plt.title("Selling Price Distribution", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=12)
plt.xlabel("Selling price in millions", weight="bold", fontsize=12)
plt.xlim(0,3000000)
plt.show()


# From the chart it is clear that the target variable is skewed

# # Most selling car in used car website ?

# In[47]:


df.car_name.value_counts()[0:10]


# Most selling used car is Hyundai i20

# In[48]:


plt.subplots(figsize=(14,7))
sns.countplot(x="car_name", data=df,ec = "black",palette="Set1",order = df['car_name'].value_counts().index)
plt.title("Top 10 Most Sold Car", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=20)
plt.xlabel("Car Name", weight="bold", fontsize=16)
plt.xticks(rotation= 45)
plt.xlim(-1,10.5)
plt.show()


# # check mean price of Hyundai i20 which is most sold

# In[50]:


i20=df[df['car_name']=='Hyundai i20']['selling_price'].mean()
i20

# The mean selling price of Hyundai i20 is 5.4lakhs
# Report 
# As per the chart these are top 10 most selling used cars.
# Of the total cars sold Hyundai i20 shares 5.8% of the total ads post and folowed by Maruti swift Dzire
# Mean Price of Most sold car is 5.4 lakhs 
# 

# # Most Selling Brand 

# In[51]:


df.brand.value_counts()[0:10]


# In[52]:


plt.subplots(figsize=(14,7))
sns.countplot(x="brand", data=df,ec = "black",palette="Set2",order = df['brand'].value_counts().index)
plt.title("Top 10 Most Sold Brand", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=14)
plt.xlabel("Brand", weight="bold", fontsize=16)
plt.xticks(rotation= 45)
plt.xlim(-1,10.5)
plt.show()


# # Check the mean Price of maruti Brand which is most sold

# In[53]:


maruti=df[df['brand']=='Maruti']['selling_price'].mean()
maruti

# Report
# As per the Chart Maruti has most share of Ads in used car website and maruti is the most sold brand
# Following Maruti we have Hyundai and Honda
# Mean selling price of Maruti Brand is 4.8lakhs
# In[54]:


brand=df.groupby('brand').selling_price.max()
brand_df=brand.to_frame().sort_values('selling_price',ascending=False)[0:10]
brand_df


# In[55]:


plt.subplots(figsize=(14,7))
sns.barplot(x=brand.index, y=brand.values,ec = "black",palette="Set2")
plt.title("Brand vs Selling Price", weight="bold",fontsize=20, pad=20)
plt.ylabel("Selling Price", weight="bold", fontsize=15)
plt.xlabel("Brand Name", weight="bold", fontsize=16)
plt.xticks(rotation=90)
plt.show()

# Report:
# Costliest brand sold is Ferrari at 3.95 cr
# Second most costliest car Brand is Rolls-Royce as 2.42cr
# Brand name has very clear impact on selling price
# # Costliest Car

# In[56]:


car=df.groupby('car_name').selling_price.max()
car=car.to_frame().sort_values('selling_price',ascending=False)[0:10]
car


# In[57]:


plt.subplots(figsize=(14,7))
sns.barplot(x=car.index, y=car.selling_price,ec = "black",palette="Set1")
plt.title("Car Name vs Selling Price", weight="bold",fontsize=20, pad=20)
plt.ylabel("Selling Price", weight="bold", fontsize=15)
plt.xlabel("Car Name", weight="bold", fontsize=16)
plt.xticks(rotation=90)
plt.show()

# Reports:
# Costliest Car Sold is Ferrari GTC4 Lusso followed by Rolls Royce Ghost
# Ferrari selling Price is 3.95 Cr
# Other than Royce other car has price below 1.5cr
# # Most Mileage Brand and Car name

# In[60]:


mileage=df.groupby('brand')['mileage'].mean().sort_values(ascending=False)
mileage.to_frame()


# In[59]:


plt.subplots(figsize=(14,7))
sns.barplot(x=mileage.index, y=mileage.values, ec = "black", palette="Set2")
plt.title("Brand vs Mileage", weight="bold",fontsize=20, pad=20)
plt.ylabel("Mileage in Kmpl", weight="bold", fontsize=15)
plt.xlabel("Brand Name", weight="bold", fontsize=12)
plt.ylim(0,25)
plt.xticks(rotation=45)
plt.show()

# Reports:
# Maruti gives the highest mileage
# Least mileage is given by Ferrari
# In[62]:


mileage_C= df.groupby('car_name')['mileage'].mean().sort_values(ascending=False).head(10)
mileage_C.to_frame()


# In[63]:


plt.subplots(figsize=(14,7))
sns.barplot(x=mileage_C.index, y=mileage_C.values, ec = "black", palette="Set1")
plt.title("Car Name vs Mileage", weight="bold",fontsize=20, pad=20)
plt.ylabel("Mileage in Kmpl", weight="bold", fontsize=15)
plt.xlabel("Car Name", weight="bold", fontsize=12)
plt.ylim(0,27)
plt.xticks(rotation=45)
plt.show()


# # Kilometer Driven vs Selling Price

# In[64]:


plt.subplots(figsize=(14,7))
sns.scatterplot(x="km_driven", y='selling_price', data=df,ec = "white",color='b', hue='fuel_type')
plt.title("Kilometer Driven vs Selling Price", weight="bold",fontsize=20, pad=20)
plt.ylabel("Selling Price", weight="bold", fontsize=20)
plt.xlim(-10000,800000) #used limit for better visualization
plt.ylim(-10000,10000000)
plt.xlabel("Kilometer driven", weight="bold", fontsize=16)
plt.show()

# Report
# Many Cars were sold with kms between 0 to 20k kms
# Low kms driven cars had more selling price compared to cars which had more kms driven
# # Fuel Type Selling Price

# In[65]:


fuel = df.groupby('fuel_type')['selling_price'].median().sort_values(ascending=False)
fuel.to_frame()


# In[66]:


plt.subplots(figsize=(14,7))
sns.barplot(x=df.fuel_type, y=df.selling_price, ec = "black", palette="Set2_r")
plt.title("Fuel type vs Selling Price", weight="bold",fontsize=20, pad=20)
plt.ylabel("Selling Price Median", weight="bold", fontsize=15)
plt.xlabel("Fuel Type", weight="bold", fontsize=12)
plt.show()

# Report
# Electric cars have higher selling avg price
# Followed by Diesel and Petrol
# Fuel Type is also important feature for the target variable
# # Most Sold Fuel Type

# In[67]:


plt.subplots(figsize=(14,7))
sns.countplot(x=df.fuel_type, ec = "black", palette="Set2_r")
plt.title("Fuel Type Count", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=15)
plt.xlabel("Fuel Type", weight="bold", fontsize=12)
plt.show()

# Report
# Petrol and Diesel dominate the used car market in the website
# The most sold fuel type is Petrol

# # Fuel types available and mileage given

# In[68]:


## mean mileage by fuel type
fuel_mileage = df.groupby('fuel_type')['mileage'].mean().sort_values(ascending=False)
fuel_mileage.to_frame()


# In[69]:


plt.subplots(figsize=(14,7))
sns.boxplot(x='fuel_type', y='mileage', data=df,palette="Set1_r")
plt.title("Fuel type vs Mileage", weight="bold",fontsize=20, pad=20)
plt.ylabel("Mileage in Kmpl", weight="bold", fontsize=15)
plt.xlabel("Fuel Type", weight="bold", fontsize=12)
plt.show()

# Report
# CNG gives the highest mileage of 25km/l
# Least mileage is given by LPG
# # # Mileage vs Selling Price

# In[70]:


plt.subplots(figsize=(14,7))
sns.scatterplot(x="mileage", y='selling_price', data=df,ec = "white",color='b', hue='fuel_type')
plt.title("Mileage vs Selling Price", weight="bold",fontsize=20, pad=20)
plt.ylabel("Selling Price", weight="bold", fontsize=20)
plt.ylim(-10000,10000000)
plt.xlabel("Mileage", weight="bold", fontsize=16)
plt.show()


# In[71]:


plt.subplots(figsize=(14,7))
sns.histplot(x=df.mileage, ec = "black", color='g', kde=True)
plt.title("Mileage Distribution", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=15)
plt.xlabel("Mileage", weight="bold", fontsize=12)
plt.show()


# # Vehicle age vs Selling Price

# In[72]:


plt.subplots(figsize=(20,10))
sns.lineplot(x='vehicle_age',y='selling_price',data=df,color='b')
plt.ylim(0,2500000)
plt.show()

# Report
# As the Vehicle age increases the price also get reduced
# Vehicle age has negative impact on selling price
# # Vehicle age vs Mileage

# In[73]:


vehicle_age = df.groupby('vehicle_age')['mileage'].median().sort_values(ascending=False)
vehicle_age.to_frame().head(5)


# # Report 
# As the age of vehicle increases the median of mileage drops
# Newer Vehicle have more mileage median older vechicle

# In[74]:


oldest = df.groupby('car_name')['vehicle_age'].max().sort_values(ascending=False).head(10)
oldest.to_frame()

# Report
# Maruti Alto is the oldest car available 29 years old in the used car website followed by BMW3 for 25 years old
# # # Transmission Type

# In[75]:


plt.subplots(figsize=(14,7))
sns.countplot(x='transmission_type', data=df,palette="Set1")
plt.title("Transmission type Count", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=15)
plt.xlabel("Transmission Type", weight="bold", fontsize=12)
plt.show() 


# In[76]:


plt.subplots(figsize=(14,7))
sns.barplot(x='transmission_type', y='selling_price', data=df,palette="Set1")
plt.title("Transmission type vs Price", weight="bold",fontsize=20, pad=20)
plt.ylabel("Selling Price in Millions", weight="bold", fontsize=15)
plt.xlabel("Transmission Type", weight="bold", fontsize=12)
plt.show() 

# Report
# Manual Transmission was found in the most of the cars which was sold
# Automatic cars have more selling price than manual cars
# # Seller Type

# In[77]:


plt.subplots(figsize=(14,7))
sns.countplot(x='seller_type', data=df,palette="rocket_r")
plt.title("Transmission type vs Price", weight="bold",fontsize=20, pad=20)
plt.ylabel("Selling Price in Millions", weight="bold", fontsize=15)
plt.xlabel("Transmission Type", weight="bold", fontsize=12)
plt.show() 


# In[78]:


dealer = df.groupby('seller_type')['selling_price'].median().sort_values(ascending=False)
dealer.to_frame()

# Report
# Dealers have put more ads on used car website
# Dealers have put 9539 ads with median selling price of 5.91 Lakhs
# Followed by individual with 5699 ads with median selling price of 5.4lakhs
# Dealer have more median selling price than individualFinal Report
# The datatypes and Column names were right and there was 15411 rows and 13 columns
# The selling_price column is the target to predict. i.e Regression Problem.
# There are outliers in the km_driven, enginer, selling_price, and max power.
# Dealers are the highest sellers of the used cars.
# Skewness is found in few of the columns will check it after handling outliers.
# Vehicle age has negative impact on the price.
# Manual cars are mostly sold and automatic has higher selling average than manual cars.
# Petrol is the most preffered choice of fuel in used car website, followed by diesel and LPG.
# We just need less data cleaning for this dataset.
# We can see from EDA that Brand and Model has same information as Carname. Hence we can drop brand and model columns and retain carname.
# # In[ ]:




