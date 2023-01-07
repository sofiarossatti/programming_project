import pandas as pd
import numpy as np
mental_health = pd.read_csv("Mental_Dataset.csv")
mental_health.info()# I see that there are many null values with different data type. I find it interesting that in the foue columns there are just 6468 values, way much less than the total. First of all, I want to change the object values (that are actually numbers) into floats.
mental_health.head() 

mental_health["Schizophrenia (%)"] = mental_health["Schizophrenia (%)"].astype(float) 
#I noticed that there is a problem in the column Schizophrenia (%). Apparentely there is a string in the column! I want to find where is located.
mystery_row=mental_health.loc[mental_health["Schizophrenia (%)"]== "Prevalence in males (%)"]
mystery_row # I find a very interesting thing: there is the possibility that my dataframe is actually made by more dataframes put together. To check that the row 6468 is the keys of a whole new dataframe, I print the following five rows just to be sure of my idea.
mental_health[6468:6473] # As I thought, this is the beginning of a whole new dataset. I have to split mental_health into parts and merge them horizontally (probably the owner has concat them vertically).

index_to_keep_0 = np.arange(6468)
mental_1 = pd.read_csv("Mental_Dataset.csv").loc[index_to_keep_0]
index_to_keep = np.arange(6469,108553)
mental_2 = pd.read_csv("Mental_Dataset.csv").loc[index_to_keep]

# now that I have two dataframe I want to clean them singularly, starting by mental_1.

# MENTAL_1
mental_1.info() # what I am interested in is checking that the number of the disorders matches with the Entity one
mental_1 = mental_1.drop(["Code", "index"], axis = 1)
mental_1["Bipolar disorder (%)"] = mental_1["Bipolar disorder (%)"].astype(float) 
mental_1["Schizophrenia (%)"] = mental_1["Schizophrenia (%)"].astype(float)
mental_1["Eating disorders (%)"] = mental_1["Eating disorders (%)"].astype(float)  
mental_1["Year"] = mental_1["Year"].astype(float)  

mental_1.info() # my data are clean now!

#MENTAL_2
mental_2.info() # Firstly, I drop the empty columns
mental_2.head()
mental_2 = mental_2.drop(["Alcohol use disorders (%)","Depression (%)", "Drug use disorders (%)", "Anxiety disorders (%)", "Code", "index"], axis = 1)
mental_2.info() # now that I have dropped the empty columns, I have to rename the remaining columns.
mental_2.rename(columns={"Schizophrenia (%)": "Prevalence in males (%)", "Bipolar disorder (%)": "Prevalence in females (%)", "Eating disorders (%)": "Population"}, inplace = True)

# Before dealing with the null values, I want to change the data that are objects in float
mental_2["Prevalence in males (%)"] = mental_2["Prevalence in males (%)"].astype(float)
# It seems that there is a string in the column. It probably means that tehre is another dataframe. I want to check the Position of the string "Suicide rate (deaths per 100,000 individuals)"
mystery_row_0= mental_2.loc[mental_2["Prevalence in males (%)"] == "Suicide rate (deaths per 100,000 individuals)"]
mystery_row_0 #starting from row 54276 there is another dataframe, so I have to split mental_2

index_to_keep1 = range(6469, 54276)
mental_2 = mental_2.loc[index_to_keep1]
mental_2 = mental_2.dropna()# Since I am interested in data from Year 1990 to 20017, I simply drop the null values
mental_2["Year"] = mental_2["Year"].astype(int)
mental_2["Prevalence in males (%)"] = mental_2["Prevalence in males (%)"].astype(float)
mental_2["Prevalence in females (%)"] = mental_2["Prevalence in females (%)"].astype(float)
mental_2["Population"] = mental_2["Population"].astype(float)

mental_2.info() #my data are clean now!

#MENTAL_3
index_to_keep2 = np.arange(54277, 108553)
mental_3 = pd.read_csv("Mental_Dataset.csv").loc[index_to_keep2]
mental_3.info()
mental_3 = mental_3.drop(["Alcohol use disorders (%)","Depression (%)", "Drug use disorders (%)", "Anxiety disorders (%)", "Entity", "Year"], axis = 1)
mental_3.rename(columns={"Schizophrenia (%)": "Suicide Rates", "Bipolar disorder (%)": "Depressive Disorder Rates", "Eating disorders (%)": "Population"}, inplace = True)

mental_3 = mental_3.drop(["Code", "index"], axis = 1) # I drop these columns since they would be a repetition in the final
mental_3['Suicide Rates'] = mental_3['Suicide Rates'].astype(float) #it seems there's another dataset, now I want to discover where it starts
mystery_row_1= mental_3.loc[mental_3["Suicide Rates"] == "Prevalence - Depressive disorders - Sex: Both - Age: All Ages (Number) (people suffering from depression)"]
mystery_row_1 # at row 102084 starts the fourth dataset, so I have to consider mental_3 as the original dataset from row 54277 to 102084

true_index = np.arange(54277,102084)
mental_3 = mental_3.loc[true_index]
mental_3 = mental_3.dropna() # Since I am interested in data from Year 1990 to 2017, I simply drop the null values (that are those from 1800 to 1989)
mental_3["Year"] = mental_3["Year"].astype(int)
mental_3["Suicide Rates "] = mental_3["Suicide Rates "].astype(float)
mental_3["Depressive Disorder Rates"] = mental_3["Depressive Disorder Rates"].astype(float)
mental_3["Population"] = mental_3["Population"].astype(float)
mental_3.info() #my data are clean now!

# MENTAL_4
final_index = np.arange(102084, 108553)
mental_4 = pd.read_csv("Mental_Dataset.csv").loc[final_index].drop(["Depression (%)", "Alcohol use disorders (%)"], axis=1 )
mental_4.rename(columns={"Schizophrenia (%)": "Prevalence", "Bipolar disorder (%)": "Depressive Disorder Rates", "Eating disorders (%)": "Sex-Both", "Anxiety disorders (%)":"All Ages", "Drug use disorders (%)": "people suffering from depression"}, inplace = True)
mental_4.info()
mental_4.head()
# Observing this dataset, I have realized that it does not offer useful data for my analysis since there are many null values and it is ambigous. Because of that, I'm not going to use it.

#Now that my 3 dataframes are clean I can merge them horizontally
mental_health_ = pd.merge(mental_1,mental_2)
mental_health_.head().T
mental_health_['Year'] = mental_health_['Year'].astype(int)
mental_health_.info()

mental_health_.index = mental_3.index
mental_health_final = pd.concat([mental_health_, mental_3], axis=1)
mental_health_final.head().T
mental_health_final["Suicide Rates"] = mental_health_final["Suicide Rates"].astype(float)
mental_health_final.info() # My final dataset is clean!

mental_health_final.rename(columns={'Entity': 'Country'}, inplace=True)

#PLOTS

# I want to analyze the disorders trend in each European state and then plot the mean of the disorders to see how they are complexly in Europe.

Austria= mental_health_final.loc[mental_health_final["Country"] == "Austria"]
Belgium = mental_health_final.loc[mental_health_final["Country"] == "Belgium"]
Bulgaria= mental_health_final.loc[mental_health_final["Country"] == "Bulgaria"]
Cyprus = mental_health_final.loc[mental_health_final["Country"] == "Cyprus"]
Croatia = mental_health_final.loc[mental_health_final["Country"] == "Croatia"]
Denmark = mental_health_final.loc[mental_health_final["Country"] == "Denmark"]
Estonia = mental_health_final.loc[mental_health_final["Country"] == "Estonia"]
Finland = mental_health_final.loc[mental_health_final["Country"] == "Finland"]
France = mental_health_final.loc[mental_health_final["Country"] == "France"]
Germany = mental_health_final.loc[mental_health_final["Country"] == "Germany"]
Greece = mental_health_final.loc[mental_health_final["Country"] == "Greece"]
Slovakia = mental_health_final.loc[mental_health_final["Country"] == "Slovakia"]
Spain = mental_health_final.loc[mental_health_final["Country"] == "Spain"]
Hungary = mental_health_final.loc[mental_health_final["Country"] == "Hungary"]
Ireland = mental_health_final.loc[mental_health_final["Country"] == "Ireland"]
Italy = mental_health_final.loc[mental_health_final["Country"] == "Italy"]
Latvia = mental_health_final.loc[mental_health_final["Country"] == "Latvia"]
Lithuania = mental_health_final.loc[mental_health_final["Country"] == "Lithuania"]
Luxembourg = mental_health_final.loc[mental_health_final["Country"] == "Luxembourg"]
Malta = mental_health_final.loc[mental_health_final["Country"] == "Malta"]
Netherlands = mental_health_final.loc[mental_health_final["Country"] == "Netherlands"]
Poland = mental_health_final.loc[mental_health_final["Country"] == "Poland"]
Portugal = mental_health_final.loc[mental_health_final["Country"] == "Portugal"]
Czech_Republic= mental_health_final.loc[mental_health_final["Country"] == "Czech Republic"]
Romania = mental_health_final.loc[mental_health_final["Country"] == "Romania"]
Slovenia = mental_health_final.loc[mental_health_final["Country"] == "Slovenia"]
Sweden = mental_health_final.loc[mental_health_final["Country"] == "Sweden"]

# STREAMLIT AND MATPLOT

import streamlit as st
import matplotlib as plt

st.header("Mental Health Project")