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
mental_health_final = mental_health_final.drop(["Population"], axis=1)
mental_health_final.head().T
mental_health_final = mental_health_final.drop(["Depressive Disorder Rates"], axis=1)
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

#MATPLOT

import matplotlib.pyplot as plt # I decided to divide the disorders into 2 plots since it is visually more clear to see the trends. I also deicded to plot separately the prevalence in gender.

import streamlit as st
#st.header("European Trends in Mental Health Project")

# AUSTRIA
years_Austria = Austria["Year"]
#st.subheader("Austria 1990-2017")
Austria_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Austria")
plt.grid(False) 
plt.axis('off')
ax_1_Au = Austria_fig1.add_subplot(2, 2, 1)
ax_2_Au = Austria_fig1.add_subplot(2, 2, 2)
ax_3_Au = Austria_fig1.add_subplot(2, 2, 3)
ax_4_Au = Austria_fig1.add_subplot(2, 2, 4)

ax_1_Au.plot(years_Austria, Austria["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Au.plot(years_Austria, Austria["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Au.plot(years_Austria, Austria["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Au.plot(years_Austria, Austria["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Au.legend()
ax_2_Au.legend()
ax_3_Au.legend()
ax_4_Au.legend()
plt.show()

Austria_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Austria")
plt.grid(False) 
plt.axis('off')
ax_5_Au = Austria_fig2.add_subplot(2, 2, 1)
ax_6_Au = Austria_fig2.add_subplot(2, 2, 2)
ax_7_Au = Austria_fig2.add_subplot(2, 2, 3)
ax_8_Au = Austria_fig2.add_subplot(2, 2, 4)

ax_5_Au.plot(years_Austria, Austria["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Au.plot(years_Austria, Austria["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Au.plot(years_Austria, Austria["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Au.plot(years_Austria, Austria["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Au.legend()
ax_6_Au.legend()
ax_7_Au.legend()
ax_8_Au.legend()
plt.show() 

Austria_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Au = Austria_fig3.add_subplot(2, 1, 1)
ax_10_Au = Austria_fig3.add_subplot(2, 1, 2)

ax_9_Au.plot(years_Austria, Austria["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Au.plot(years_Austria, Austria["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Au.legend()
ax_10_Au.legend()
plt.show()


# BELGIUM

years_Belgium = Belgium["Year"]

Belgium_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Belgium")
plt.grid(False) 
plt.axis('off')
ax_1_Be = Belgium_fig1.add_subplot(2, 2, 1)
ax_2_Be = Belgium_fig1.add_subplot(2, 2, 2)
ax_3_Be = Belgium_fig1.add_subplot(2, 2, 3)
ax_4_Be = Belgium_fig1.add_subplot(2, 2, 4)

ax_1_Be.plot(years_Belgium, Belgium["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Be.plot(years_Belgium, Belgium["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Be.plot(years_Belgium, Belgium["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Be.plot(years_Belgium, Belgium["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Be.legend()
ax_2_Be.legend()
ax_3_Be.legend()
ax_4_Be.legend()
plt.show()

Belgium_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Belgium")
plt.grid(False) 
plt.axis('off')
ax_5_Be = Belgium_fig2.add_subplot(2, 2, 1)
ax_6_Be = Belgium_fig2.add_subplot(2, 2, 2)
ax_7_Be = Belgium_fig2.add_subplot(2, 2, 3)
ax_8_Be = Belgium_fig2.add_subplot(2, 2, 4)

ax_5_Be.plot(years_Belgium, Belgium["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Be.plot(years_Belgium, Belgium["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Be.plot(years_Belgium, Belgium["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Be.plot(years_Belgium, Belgium["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Be.legend()
ax_6_Be.legend()
ax_7_Be.legend()
ax_8_Be.legend()
plt.show() 

Belgium_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Be = Belgium_fig3.add_subplot(2, 1, 1)
ax_10_Be = Belgium_fig3.add_subplot(2, 1, 2)

ax_9_Be.plot(years_Belgium, Belgium["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Be.plot(years_Belgium, Belgium["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Be.legend()
ax_10_Be.legend()
plt.show()

# BULGARIA

years_Bulgaria = Bulgaria["Year"]

Bulgaria_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Bulgaria")
plt.grid(False) 
plt.axis('off')
ax_1_Bu = Bulgaria_fig1.add_subplot(2, 2, 1)
ax_2_Bu = Bulgaria_fig1.add_subplot(2, 2, 2)
ax_3_Bu = Bulgaria_fig1.add_subplot(2, 2, 3)
ax_4_Bu = Bulgaria_fig1.add_subplot(2, 2, 4)

ax_1_Bu.plot(years_Bulgaria, Bulgaria["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Bu.plot(years_Bulgaria, Bulgaria["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Bu.plot(years_Bulgaria, Bulgaria["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Bu.plot(years_Bulgaria, Bulgaria["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Bu.legend()
ax_2_Bu.legend()
ax_3_Bu.legend()
ax_4_Bu.legend()
plt.show()

Bulgaria_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Bulgaria")
plt.grid(False) 
plt.axis('off')
ax_5_Bu = Bulgaria_fig2.add_subplot(2, 2, 1)
ax_6_Bu = Bulgaria_fig2.add_subplot(2, 2, 2)
ax_7_Bu = Bulgaria_fig2.add_subplot(2, 2, 3)
ax_8_Bu = Bulgaria_fig2.add_subplot(2, 2, 4)

ax_5_Bu.plot(years_Bulgaria, Bulgaria["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Bu.plot(years_Bulgaria, Bulgaria["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Bu.plot(years_Bulgaria, Bulgaria["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Bu.plot(years_Bulgaria, Bulgaria["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Bu.legend()
ax_6_Bu.legend()
ax_7_Bu.legend()
ax_8_Bu.legend()
plt.show() 

Bulgaria_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Bu = Bulgaria_fig3.add_subplot(2, 1, 1)
ax_10_Bu = Bulgaria_fig3.add_subplot(2, 1, 2)

ax_9_Bu.plot(years_Bulgaria, Bulgaria["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Bu.plot(years_Bulgaria, Bulgaria["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Bu.legend()
ax_10_Bu.legend()
plt.show()

#CYPRUS

years_Cyprus = Cyprus["Year"]

Cyprus_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Cyprus")
plt.grid(False) 
plt.axis('off')
ax_1_Cy = Cyprus_fig1.add_subplot(2, 2, 1)
ax_2_Cy = Cyprus_fig1.add_subplot(2, 2, 2)
ax_3_Cy = Cyprus_fig1.add_subplot(2, 2, 3)
ax_4_Cy = Cyprus_fig1.add_subplot(2, 2, 4)

ax_1_Cy.plot(years_Cyprus, Cyprus["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Cy.plot(years_Cyprus, Cyprus["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Cy.plot(years_Cyprus, Cyprus["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Cy.plot(years_Cyprus, Cyprus["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Cy.legend()
ax_2_Cy.legend()
ax_3_Cy.legend()
ax_4_Cy.legend()
plt.show()

Cyprus_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Cyprus")
plt.grid(False) 
plt.axis('off')
ax_5_Cy = Cyprus_fig2.add_subplot(2, 2, 1)
ax_6_Cy = Cyprus_fig2.add_subplot(2, 2, 2)
ax_7_Cy = Cyprus_fig2.add_subplot(2, 2, 3)
ax_8_Cy = Cyprus_fig2.add_subplot(2, 2, 4)

ax_5_Cy.plot(years_Cyprus, Cyprus["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Cy.plot(years_Cyprus, Cyprus["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Cy.plot(years_Cyprus, Cyprus["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Cy.plot(years_Cyprus, Cyprus["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Cy.legend()
ax_6_Cy.legend()
ax_7_Cy.legend()
ax_8_Cy.legend()
plt.show() 

Cyprus_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Cy = Cyprus_fig3.add_subplot(2, 1, 1)
ax_10_Cy = Cyprus_fig3.add_subplot(2, 1, 2)

ax_9_Cy.plot(years_Cyprus, Cyprus["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Cy.plot(years_Cyprus, Cyprus["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Cy.legend()
ax_10_Cy.legend()
plt.show()

#CROATIA

years_Croatia = Croatia["Year"]

Croatia_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Croatia")
plt.grid(False) 
plt.axis('off')
ax_1_Cro = Croatia_fig1.add_subplot(2, 2, 1)
ax_2_Cro = Croatia_fig1.add_subplot(2, 2, 2)
ax_3_Cro = Croatia_fig1.add_subplot(2, 2, 3)
ax_4_Cro = Croatia_fig1.add_subplot(2, 2, 4)

ax_1_Cro.plot(years_Croatia, Croatia["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Cro.plot(years_Croatia, Croatia["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Cro.plot(years_Croatia, Croatia["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Cro.plot(years_Croatia, Croatia["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Cro.legend()
ax_2_Cro.legend()
ax_3_Cro.legend()
ax_4_Cro.legend()
plt.show()

Croatia_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Croatia")
plt.grid(False) 
plt.axis('off')
ax_5_Cro = Croatia_fig2.add_subplot(2, 2, 1)
ax_6_Cro = Croatia_fig2.add_subplot(2, 2, 2)
ax_7_Cro = Croatia_fig2.add_subplot(2, 2, 3)
ax_8_Cro = Croatia_fig2.add_subplot(2, 2, 4)

ax_5_Cro.plot(years_Croatia, Croatia["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Cro.plot(years_Croatia, Croatia["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Cro.plot(years_Croatia, Croatia["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Cro.plot(years_Croatia, Croatia["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Cro.legend()
ax_6_Cro.legend()
ax_7_Cro.legend()
ax_8_Cro.legend()
plt.show() 

Croatia_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Cro = Croatia_fig3.add_subplot(2, 1, 1)
ax_10_Cro = Croatia_fig3.add_subplot(2, 1, 2)

ax_9_Cro.plot(years_Croatia, Croatia["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Cro.plot(years_Croatia, Croatia["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Cro.legend()
ax_10_Cro.legend()
plt.show()

# DENMARK

years_Denmark = Denmark["Year"]

Denmark_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Denmark")
plt.grid(False) 
plt.axis('off')
ax_1_Den = Denmark_fig1.add_subplot(2, 2, 1)
ax_2_Den = Denmark_fig1.add_subplot(2, 2, 2)
ax_3_Den = Denmark_fig1.add_subplot(2, 2, 3)
ax_4_Den = Denmark_fig1.add_subplot(2, 2, 4)

ax_1_Den.plot(years_Denmark, Denmark["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Den.plot(years_Denmark, Denmark["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Den.plot(years_Denmark, Denmark["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Den.plot(years_Denmark, Denmark["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Den.legend()
ax_2_Den.legend()
ax_3_Den.legend()
ax_4_Den.legend()
plt.show()

Denmark_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Denmark")
plt.grid(False) 
plt.axis('off')
ax_5_Den = Denmark_fig2.add_subplot(2, 2, 1)
ax_6_Den = Denmark_fig2.add_subplot(2, 2, 2)
ax_7_Den = Denmark_fig2.add_subplot(2, 2, 3)
ax_8_Den = Denmark_fig2.add_subplot(2, 2, 4)

ax_5_Den.plot(years_Denmark, Denmark["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Den.plot(years_Denmark, Denmark["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Den.plot(years_Denmark, Denmark["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Den.plot(years_Denmark, Denmark["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Den.legend()
ax_6_Den.legend()
ax_7_Den.legend()
ax_8_Den.legend()
plt.show() 

Denmark_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Den = Denmark_fig3.add_subplot(2, 1, 1)
ax_10_Den = Denmark_fig3.add_subplot(2, 1, 2)

ax_9_Den.plot(years_Denmark, Denmark["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Den.plot(years_Denmark, Denmark["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Den.legend()
ax_10_Den.legend()
plt.show()

# ESTONIA

years_Estonia = Estonia["Year"]
Estonia_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Estonia")
plt.grid(False) 
plt.axis('off')

ax_1_Est = Estonia_fig1.add_subplot(2, 2, 1)
ax_2_Est = Estonia_fig1.add_subplot(2, 2, 2)
ax_3_Est = Estonia_fig1.add_subplot(2, 2, 3)
ax_4_Est = Estonia_fig1.add_subplot(2, 2, 4)

ax_1_Est.plot(years_Estonia,Estonia["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Est.plot(years_Estonia,Estonia["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Est.plot(years_Estonia,Estonia["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Est.plot(years_Estonia,Estonia["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Est.legend()
ax_2_Est.legend()
ax_3_Est.legend()
ax_4_Est.legend()
plt.show()

Estonia_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Estonia")
plt.grid(False) 
plt.axis('off')
ax_5_Est = Estonia_fig2.add_subplot(2, 2, 1)
ax_6_Est = Estonia_fig2.add_subplot(2, 2, 2)
ax_7_Est = Estonia_fig2.add_subplot(2, 2, 3)
ax_8_Est = Estonia_fig2.add_subplot(2, 2, 4)

ax_5_Est.plot(years_Estonia,Estonia["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Est.plot(years_Estonia,Estonia["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Est.plot(years_Estonia,Estonia["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Est.plot(years_Estonia,Estonia["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Est.legend()
ax_6_Est.legend()
ax_7_Est.legend()
ax_8_Est.legend()
plt.show() 

Estonia_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Est = Estonia_fig3.add_subplot(2, 1, 1)
ax_10_Est = Estonia_fig3.add_subplot(2, 1, 2)

ax_9_Est.plot(years_Estonia,Estonia["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Est.plot(years_Estonia,Estonia["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Est.legend()
ax_10_Est.legend()
plt.show()

# FINLAND

years_Finland = Finland["Year"]
Finland_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Finland")
plt.grid(False) 
plt.axis('off')

ax_1_Fin = Finland_fig1.add_subplot(2, 2, 1)
ax_2_Fin = Finland_fig1.add_subplot(2, 2, 2)
ax_3_Fin = Finland_fig1.add_subplot(2, 2, 3)
ax_4_Fin = Finland_fig1.add_subplot(2, 2, 4)

ax_1_Fin.plot(years_Finland,Finland["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Fin.plot(years_Finland,Finland["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Fin.plot(years_Finland,Finland["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Fin.plot(years_Finland,Finland["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Fin.legend()
ax_2_Fin.legend()
ax_3_Fin.legend()
ax_4_Fin.legend()
plt.show()

Finland_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Finland")
plt.grid(False) 
plt.axis('off')
ax_5_Fin = Finland_fig2.add_subplot(2, 2, 1)
ax_6_Fin = Finland_fig2.add_subplot(2, 2, 2)
ax_7_Fin = Finland_fig2.add_subplot(2, 2, 3)
ax_8_Fin = Finland_fig2.add_subplot(2, 2, 4)

ax_5_Fin.plot(years_Finland,Finland["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Fin.plot(years_Finland,Finland["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Fin.plot(years_Finland,Finland["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Fin.plot(years_Finland,Finland["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Fin.legend()
ax_6_Fin.legend()
ax_7_Fin.legend()
ax_8_Fin.legend()
plt.show() 

Finland_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Fin = Finland_fig3.add_subplot(2, 1, 1)
ax_10_Fin = Finland_fig3.add_subplot(2, 1, 2)

ax_9_Fin.plot(years_Finland,Finland["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Fin.plot(years_Finland,Finland["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Fin.legend()
ax_10_Fin.legend()
plt.show()

# FRANCE

years_France = France["Year"]
France_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across France")
plt.grid(False) 
plt.axis('off')

ax_1_Fr = France_fig1.add_subplot(2, 2, 1)
ax_2_Fr = France_fig1.add_subplot(2, 2, 2)
ax_3_Fr = France_fig1.add_subplot(2, 2, 3)
ax_4_Fr = France_fig1.add_subplot(2, 2, 4)

ax_1_Fr.plot(years_France,France["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Fr.plot(years_France,France["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Fr.plot(years_France,France["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Fr.plot(years_France,France["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Fr.legend()
ax_2_Fr.legend()
ax_3_Fr.legend()
ax_4_Fr.legend()
plt.show()

France_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across France")
plt.grid(False) 
plt.axis('off')
ax_5_Fr = France_fig2.add_subplot(2, 2, 1)
ax_6_Fr = France_fig2.add_subplot(2, 2, 2)
ax_7_Fr = France_fig2.add_subplot(2, 2, 3)
ax_8_Fr = France_fig2.add_subplot(2, 2, 4)

ax_5_Fr.plot(years_France,France["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Fr.plot(years_France,France["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Fr.plot(years_France,France["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Fr.plot(years_France,France["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Fr.legend()
ax_6_Fr.legend()
ax_7_Fr.legend()
ax_8_Fr.legend()
plt.show() 

France_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Fr = France_fig3.add_subplot(2, 1, 1)
ax_10_Fr = France_fig3.add_subplot(2, 1, 2)

ax_9_Fr.plot(years_France,France["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Fr.plot(years_France,France["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Fr.legend()
ax_10_Fr.legend()
plt.show()

# GERMANY

years_Germany = Germany["Year"]
Germany_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Germany")
plt.grid(False) 
plt.axis('off')

ax_1_Ge = Germany_fig1.add_subplot(2, 2, 1)
ax_2_Ge = Germany_fig1.add_subplot(2, 2, 2)
ax_3_Ge = Germany_fig1.add_subplot(2, 2, 3)
ax_4_Ge = Germany_fig1.add_subplot(2, 2, 4)

ax_1_Ge.plot(years_Germany,Germany["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Ge.plot(years_Germany,Germany["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Ge.plot(years_Germany,Germany["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Ge.plot(years_Germany,Germany["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Ge.legend()
ax_2_Ge.legend()
ax_3_Ge.legend()
ax_4_Ge.legend()
plt.show()

Germany_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Germany")
plt.grid(False) 
plt.axis('off')
ax_5_Ge = Germany_fig2.add_subplot(2, 2, 1)
ax_6_Ge = Germany_fig2.add_subplot(2, 2, 2)
ax_7_Ge = Germany_fig2.add_subplot(2, 2, 3)
ax_8_Ge = Germany_fig2.add_subplot(2, 2, 4)

ax_5_Ge.plot(years_Germany,Germany["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Ge.plot(years_Germany,Germany["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Ge.plot(years_Germany,Germany["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Ge.plot(years_Germany,Germany["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Ge.legend()
ax_6_Ge.legend()
ax_7_Ge.legend()
ax_8_Ge.legend()
plt.show() 

Germany_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Ge = Germany_fig3.add_subplot(2, 1, 1)
ax_10_Ge = Germany_fig3.add_subplot(2, 1, 2)

ax_9_Ge.plot(years_Germany,Germany["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Ge.plot(years_Germany,Germany["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Ge.legend()
ax_10_Ge.legend()
plt.show()

# GREECE

years_Greece = Greece["Year"]
Greece_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Greece")
plt.grid(False) 
plt.axis('off')

ax_1_Gr = Greece_fig1.add_subplot(2, 2, 1)
ax_2_Gr = Greece_fig1.add_subplot(2, 2, 2)
ax_3_Gr = Greece_fig1.add_subplot(2, 2, 3)
ax_4_Gr = Greece_fig1.add_subplot(2, 2, 4)

ax_1_Gr.plot(years_Greece,Greece["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Gr.plot(years_Greece,Greece["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Gr.plot(years_Greece,Greece["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Gr.plot(years_Greece,Greece["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Gr.legend()
ax_2_Gr.legend()
ax_3_Gr.legend()
ax_4_Gr.legend()
plt.show()

Greece_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Greece")
plt.grid(False) 
plt.axis('off')
ax_5_Gr = Greece_fig2.add_subplot(2, 2, 1)
ax_6_Gr = Greece_fig2.add_subplot(2, 2, 2)
ax_7_Gr = Greece_fig2.add_subplot(2, 2, 3)
ax_8_Gr = Greece_fig2.add_subplot(2, 2, 4)

ax_5_Gr.plot(years_Greece,Greece["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Gr.plot(years_Greece,Greece["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Gr.plot(years_Greece,Greece["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Gr.plot(years_Greece,Greece["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Gr.legend()
ax_6_Gr.legend()
ax_7_Gr.legend()
ax_8_Gr.legend()
plt.show() 

Greece_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Gr = Greece_fig3.add_subplot(2, 1, 1)
ax_10_Gr = Greece_fig3.add_subplot(2, 1, 2)

ax_9_Gr.plot(years_Greece,Greece["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Gr.plot(years_Greece,Greece["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Gr.legend()
ax_10_Gr.legend()
plt.show()

# SLOVAKIA

years_Slovakia = Slovakia["Year"]
Slovakia_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Slovakia")
plt.grid(False) 
plt.axis('off')

ax_1_Slok = Slovakia_fig1.add_subplot(2, 2, 1)
ax_2_Slok = Slovakia_fig1.add_subplot(2, 2, 2)
ax_3_Slok = Slovakia_fig1.add_subplot(2, 2, 3)
ax_4_Slok = Slovakia_fig1.add_subplot(2, 2, 4)

ax_1_Slok.plot(years_Slovakia,Slovakia["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Slok.plot(years_Slovakia,Slovakia["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Slok.plot(years_Slovakia,Slovakia["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Slok.plot(years_Slovakia,Slovakia["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Slok.legend()
ax_2_Slok.legend()
ax_3_Slok.legend()
ax_4_Slok.legend()
plt.show()

Slovakia_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Slovakia")
plt.grid(False) 
plt.axis('off')
ax_5_Slok = Slovakia_fig2.add_subplot(2, 2, 1)
ax_6_Slok = Slovakia_fig2.add_subplot(2, 2, 2)
ax_7_Slok = Slovakia_fig2.add_subplot(2, 2, 3)
ax_8_Slok = Slovakia_fig2.add_subplot(2, 2, 4)

ax_5_Slok.plot(years_Slovakia,Slovakia["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Slok.plot(years_Slovakia,Slovakia["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Slok.plot(years_Slovakia,Slovakia["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Slok.plot(years_Slovakia,Slovakia["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Slok.legend()
ax_6_Slok.legend()
ax_7_Slok.legend()
ax_8_Slok.legend()
plt.show() 

Slovakia_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Slok = Slovakia_fig3.add_subplot(2, 1, 1)
ax_10_Slok = Slovakia_fig3.add_subplot(2, 1, 2)

ax_9_Slok.plot(years_Slovakia,Slovakia["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Slok.plot(years_Slovakia,Slovakia["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Slok.legend()
ax_10_Slok.legend()
plt.show()

# SPAIN

years_Spain = Spain["Year"]
Spain_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Spain")
plt.grid(False) 
plt.axis('off')

ax_1_Spa = Spain_fig1.add_subplot(2, 2, 1)
ax_2_Spa = Spain_fig1.add_subplot(2, 2, 2)
ax_3_Spa = Spain_fig1.add_subplot(2, 2, 3)
ax_4_Spa = Spain_fig1.add_subplot(2, 2, 4)

ax_1_Spa.plot(years_Spain,Spain["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Spa.plot(years_Spain,Spain["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Spa.plot(years_Spain,Spain["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Spa.plot(years_Spain,Spain["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Spa.legend()
ax_2_Spa.legend()
ax_3_Spa.legend()
ax_4_Spa.legend()
plt.show()

Spain_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Spain")
plt.grid(False) 
plt.axis('off')
ax_5_Spa = Spain_fig2.add_subplot(2, 2, 1)
ax_6_Spa = Spain_fig2.add_subplot(2, 2, 2)
ax_7_Spa = Spain_fig2.add_subplot(2, 2, 3)
ax_8_Spa = Spain_fig2.add_subplot(2, 2, 4)

ax_5_Spa.plot(years_Spain,Spain["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Spa.plot(years_Spain,Spain["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Spa.plot(years_Spain,Spain["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Spa.plot(years_Spain,Spain["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Spa.legend()
ax_6_Spa.legend()
ax_7_Spa.legend()
ax_8_Spa.legend()
plt.show() 

Spain_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Spa = Spain_fig3.add_subplot(2, 1, 1)
ax_10_Spa = Spain_fig3.add_subplot(2, 1, 2)

ax_9_Spa.plot(years_Spain,Spain["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Spa.plot(years_Spain,Spain["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Spa.legend()
ax_10_Spa.legend()
plt.show()

# HUNGARY

years_Hungary = Hungary["Year"]
Hungary_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Hungary")
plt.grid(False) 
plt.axis('off')

ax_1_Hu = Hungary_fig1.add_subplot(2, 2, 1)
ax_2_Hu = Hungary_fig1.add_subplot(2, 2, 2)
ax_3_Hu = Hungary_fig1.add_subplot(2, 2, 3)
ax_4_Hu = Hungary_fig1.add_subplot(2, 2, 4)

ax_1_Hu.plot(years_Hungary,Hungary["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Hu.plot(years_Hungary,Hungary["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Hu.plot(years_Hungary,Hungary["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Hu.plot(years_Hungary,Hungary["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Hu.legend()
ax_2_Hu.legend()
ax_3_Hu.legend()
ax_4_Hu.legend()
plt.show()

Hungary_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Hungary")
plt.grid(False) 
plt.axis('off')
ax_5_Hu = Hungary_fig2.add_subplot(2, 2, 1)
ax_6_Hu = Hungary_fig2.add_subplot(2, 2, 2)
ax_7_Hu = Hungary_fig2.add_subplot(2, 2, 3)
ax_8_Hu = Hungary_fig2.add_subplot(2, 2, 4)

ax_5_Hu.plot(years_Hungary,Hungary["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Hu.plot(years_Hungary,Hungary["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Hu.plot(years_Hungary,Hungary["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Hu.plot(years_Hungary,Hungary["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Hu.legend()
ax_6_Hu.legend()
ax_7_Hu.legend()
ax_8_Hu.legend()
plt.show() 

Hungary_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Hu = Hungary_fig3.add_subplot(2, 1, 1)
ax_10_Hu = Hungary_fig3.add_subplot(2, 1, 2)

ax_9_Hu.plot(years_Hungary,Hungary["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Hu.plot(years_Hungary,Hungary["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Hu.legend()
ax_10_Hu.legend()
plt.show()

# IRLANDA

years_Ireland = Ireland["Year"]
Ireland_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Ireland")
plt.grid(False) 
plt.axis('off')

ax_1 = Ireland_fig1.add_subplot(2, 2, 1)
ax_2 = Ireland_fig1.add_subplot(2, 2, 2)
ax_3 = Ireland_fig1.add_subplot(2, 2, 3)
ax_4 = Ireland_fig1.add_subplot(2, 2, 4)

ax_1.plot(years_Ireland,Ireland["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2.plot(years_Ireland,Ireland["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3.plot(years_Ireland,Ireland["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4.plot(years_Ireland,Ireland["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1.legend()
ax_2.legend()
ax_3.legend()
ax_4.legend()
plt.show()

Ireland_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Ireland")
plt.grid(False) 
plt.axis('off')
ax_5 = Ireland_fig2.add_subplot(2, 2, 1)
ax_6 = Ireland_fig2.add_subplot(2, 2, 2)
ax_7 = Ireland_fig2.add_subplot(2, 2, 3)
ax_8 = Ireland_fig2.add_subplot(2, 2, 4)

ax_5.plot(years_Ireland,Ireland["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6.plot(years_Ireland,Ireland["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7.plot(years_Ireland,Ireland["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8.plot(years_Ireland,Ireland["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5.legend()
ax_6.legend()
ax_7.legend()
ax_8.legend()
plt.show() 

Ireland_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9 = Ireland_fig3.add_subplot(2, 1, 1)
ax_10 = Ireland_fig3.add_subplot(2, 1, 2)

ax_9.plot(years_Ireland,Ireland["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10.plot(years_Ireland,Ireland["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9.legend()
ax_10.legend()
plt.show()

#ITALY

years_Italy = Italy["Year"]
Italy_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Italy")
plt.grid(False) 
plt.axis('off')

ax_1_It = Italy_fig1.add_subplot(2, 2, 1)
ax_2_It = Italy_fig1.add_subplot(2, 2, 2)
ax_3_It = Italy_fig1.add_subplot(2, 2, 3)
ax_4_It = Italy_fig1.add_subplot(2, 2, 4)

ax_1_It.plot(years_Italy,Italy["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_It.plot(years_Italy,Italy["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_It.plot(years_Italy,Italy["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_It.plot(years_Italy,Italy["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_It.legend()
ax_2_It.legend()
ax_3_It.legend()
ax_4_It.legend()
plt.show()

Italy_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Italy")
plt.grid(False) 
plt.axis('off')
ax_5_It = Italy_fig2.add_subplot(2, 2, 1)
ax_6_It = Italy_fig2.add_subplot(2, 2, 2)
ax_7_It = Italy_fig2.add_subplot(2, 2, 3)
ax_8_It = Italy_fig2.add_subplot(2, 2, 4)

ax_5_It.plot(years_Italy,Italy["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_It.plot(years_Italy,Italy["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_It.plot(years_Italy,Italy["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_It.plot(years_Italy,Italy["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_It.legend()
ax_6_It.legend()
ax_7_It.legend()
ax_8_It.legend()
plt.show() 

Italy_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_It = Italy_fig3.add_subplot(2, 1, 1)
ax_10_It = Italy_fig3.add_subplot(2, 1, 2)

ax_9_It.plot(years_Italy,Italy["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_It.plot(years_Italy,Italy["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_It.legend()
ax_10_It.legend()
plt.show()

# LETTONIA

years_Latvia = Latvia["Year"]
Latvia_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Latvia")
plt.grid(False) 
plt.axis('off')

ax_1_Let = Latvia_fig1.add_subplot(2, 2, 1)
ax_2_Let = Latvia_fig1.add_subplot(2, 2, 2)
ax_3_Let = Latvia_fig1.add_subplot(2, 2, 3)
ax_4_Let = Latvia_fig1.add_subplot(2, 2, 4)

ax_1_Let.plot(years_Latvia,Latvia["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Let.plot(years_Latvia,Latvia["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Let.plot(years_Latvia,Latvia["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Let.plot(years_Latvia,Latvia["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Let.legend()
ax_2_Let.legend()
ax_3_Let.legend()
ax_4_Let.legend()
plt.show()

Latvia_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Latvia")
plt.grid(False) 
plt.axis('off')
ax_5_Let = Latvia_fig2.add_subplot(2, 2, 1)
ax_6_Let = Latvia_fig2.add_subplot(2, 2, 2)
ax_7_Let = Latvia_fig2.add_subplot(2, 2, 3)
ax_8_Let = Latvia_fig2.add_subplot(2, 2, 4)

ax_5_Let.plot(years_Latvia,Latvia["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Let.plot(years_Latvia,Latvia["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Let.plot(years_Latvia,Latvia["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Let.plot(years_Latvia,Latvia["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Let.legend()
ax_6_Let.legend()
ax_7_Let.legend()
ax_8_Let.legend()
plt.show() 

Latvia_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Let = Latvia_fig3.add_subplot(2, 1, 1)
ax_10_Let = Latvia_fig3.add_subplot(2, 1, 2)

ax_9_Let.plot(years_Latvia,Latvia["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Let.plot(years_Latvia,Latvia["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Let.legend()
ax_10_Let.legend()
plt.show()

# LITHUANIA

years_Lithuania = Lithuania["Year"]
Lithuania_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Lithuania")
plt.grid(False) 
plt.axis('off')

ax_1_Lit = Lithuania_fig1.add_subplot(2, 2, 1)
ax_2_Lit = Lithuania_fig1.add_subplot(2, 2, 2)
ax_3_Lit = Lithuania_fig1.add_subplot(2, 2, 3)
ax_4_Lit = Lithuania_fig1.add_subplot(2, 2, 4)

ax_1_Lit.plot(years_Lithuania,Lithuania["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Lit.plot(years_Lithuania,Lithuania["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Lit.plot(years_Lithuania,Lithuania["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Lit.plot(years_Lithuania,Lithuania["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Lit.legend()
ax_2_Lit.legend()
ax_3_Lit.legend()
ax_4_Lit.legend()
plt.show()

Lithuania_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Lithuania")
plt.grid(False) 
plt.axis('off')
ax_5_Lit = Lithuania_fig2.add_subplot(2, 2, 1)
ax_6_Lit = Lithuania_fig2.add_subplot(2, 2, 2)
ax_7_Lit = Lithuania_fig2.add_subplot(2, 2, 3)
ax_8_Lit = Lithuania_fig2.add_subplot(2, 2, 4)

ax_5_Lit.plot(years_Lithuania,Lithuania["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Lit.plot(years_Lithuania,Lithuania["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Lit.plot(years_Lithuania,Lithuania["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Lit.plot(years_Lithuania,Lithuania["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Lit.legend()
ax_6_Lit.legend()
ax_7_Lit.legend()
ax_8_Lit.legend()
plt.show() 

Lithuania_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Lit = Lithuania_fig3.add_subplot(2, 1, 1)
ax_10_Lit = Lithuania_fig3.add_subplot(2, 1, 2)

ax_9_Lit.plot(years_Lithuania,Lithuania["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Lit.plot(years_Lithuania,Lithuania["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Lit.legend()
ax_10_Lit.legend()
plt.show()

# LUXEMBOURG

years_Luxembourg = Luxembourg["Year"]
Luxembourg_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Luxembourg")
plt.grid(False) 
plt.axis('off')

ax_1_Lux = Luxembourg_fig1.add_subplot(2, 2, 1)
ax_2_Lux = Luxembourg_fig1.add_subplot(2, 2, 2)
ax_3_Lux = Luxembourg_fig1.add_subplot(2, 2, 3)
ax_4_Lux = Luxembourg_fig1.add_subplot(2, 2, 4)

ax_1_Lux.plot(years_Luxembourg,Luxembourg["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Lux.plot(years_Luxembourg,Luxembourg["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Lux.plot(years_Luxembourg,Luxembourg["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Lux.plot(years_Luxembourg,Luxembourg["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Lux.legend()
ax_2_Lux.legend()
ax_3_Lux.legend()
ax_4_Lux.legend()
plt.show()

Luxembourg_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Luxembourg")
plt.grid(False) 
plt.axis('off')
ax_5_Lux = Luxembourg_fig2.add_subplot(2, 2, 1)
ax_6_Lux = Luxembourg_fig2.add_subplot(2, 2, 2)
ax_7_Lux = Luxembourg_fig2.add_subplot(2, 2, 3)
ax_8_Lux = Luxembourg_fig2.add_subplot(2, 2, 4)

ax_5_Lux.plot(years_Luxembourg,Luxembourg["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Lux.plot(years_Luxembourg,Luxembourg["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Lux.plot(years_Luxembourg,Luxembourg["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Lux.plot(years_Luxembourg,Luxembourg["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Lux.legend()
ax_6_Lux.legend()
ax_7_Lux.legend()
ax_8_Lux.legend()
plt.show() 

Luxembourg_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Lux = Luxembourg_fig3.add_subplot(2, 1, 1)
ax_10_Lux = Luxembourg_fig3.add_subplot(2, 1, 2)

ax_9_Lux.plot(years_Luxembourg,Luxembourg["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Lux.plot(years_Luxembourg,Luxembourg["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Lux.legend()
ax_10_Lux.legend()
plt.show()

# MALTA

years_Malta = Malta["Year"]
Malta_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Malta")
plt.grid(False) 
plt.axis('off')

ax_1_Ma = Malta_fig1.add_subplot(2, 2, 1)
ax_2_Ma = Malta_fig1.add_subplot(2, 2, 2)
ax_3_Ma = Malta_fig1.add_subplot(2, 2, 3)
ax_4_Ma = Malta_fig1.add_subplot(2, 2, 4)

ax_1_Ma.plot(years_Malta,Malta["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Ma.plot(years_Malta,Malta["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Ma.plot(years_Malta,Malta["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Ma.plot(years_Malta,Malta["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Ma.legend()
ax_2_Ma.legend()
ax_3_Ma.legend()
ax_4_Ma.legend()
plt.show()

Malta_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Malta")
plt.grid(False) 
plt.axis('off')
ax_5_Ma = Malta_fig2.add_subplot(2, 2, 1)
ax_6_Ma = Malta_fig2.add_subplot(2, 2, 2)
ax_7_Ma = Malta_fig2.add_subplot(2, 2, 3)
ax_8_Ma = Malta_fig2.add_subplot(2, 2, 4)

ax_5_Ma.plot(years_Malta,Malta["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Ma.plot(years_Malta,Malta["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Ma.plot(years_Malta,Malta["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Ma.plot(years_Malta,Malta["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Ma.legend()
ax_6_Ma.legend()
ax_7_Ma.legend()
ax_8_Ma.legend()
plt.show() 

Malta_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Ma = Malta_fig3.add_subplot(2, 1, 1)
ax_10_Ma = Malta_fig3.add_subplot(2, 1, 2)

ax_9_Ma.plot(years_Malta,Malta["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Ma.plot(years_Malta,Malta["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Ma.legend()
ax_10_Ma.legend()
plt.show()

# HOLLAND

years_Netherlands = Netherlands["Year"]
Netherlands_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Netherlands")
plt.grid(False) 
plt.axis('off')

ax_1_Ho = Netherlands_fig1.add_subplot(2, 2, 1)
ax_2_Ho = Netherlands_fig1.add_subplot(2, 2, 2)
ax_3_Ho = Netherlands_fig1.add_subplot(2, 2, 3)
ax_4_Ho = Netherlands_fig1.add_subplot(2, 2, 4)

ax_1_Ho.plot(years_Netherlands,Netherlands["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Ho.plot(years_Netherlands,Netherlands["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Ho.plot(years_Netherlands,Netherlands["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Ho.plot(years_Netherlands,Netherlands["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Ho.legend()
ax_2_Ho.legend()
ax_3_Ho.legend()
ax_4_Ho.legend()
plt.show()

Netherlands_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Netherlands")
plt.grid(False) 
plt.axis('off')
ax_5_Ho = Netherlands_fig2.add_subplot(2, 2, 1)
ax_6_Ho = Netherlands_fig2.add_subplot(2, 2, 2)
ax_7_Ho = Netherlands_fig2.add_subplot(2, 2, 3)
ax_8_Ho = Netherlands_fig2.add_subplot(2, 2, 4)

ax_5_Ho.plot(years_Netherlands,Netherlands["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Ho.plot(years_Netherlands,Netherlands["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Ho.plot(years_Netherlands,Netherlands["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Ho.plot(years_Netherlands,Netherlands["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Ho.legend()
ax_6_Ho.legend()
ax_7_Ho.legend()
ax_8_Ho.legend()
plt.show() 

Netherlands_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Ho = Netherlands_fig3.add_subplot(2, 1, 1)
ax_10_Ho = Netherlands_fig3.add_subplot(2, 1, 2)

ax_9_Ho.plot(years_Netherlands,Netherlands["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Ho.plot(years_Netherlands,Netherlands["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Ho.legend()
ax_10_Ho.legend()
plt.show()

# POLAND

years_Poland = Poland["Year"]
Poland_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Poland")
plt.grid(False) 
plt.axis('off')

ax_1_Pol = Poland_fig1.add_subplot(2, 2, 1)
ax_2_Pol = Poland_fig1.add_subplot(2, 2, 2)
ax_3_Pol = Poland_fig1.add_subplot(2, 2, 3)
ax_4_Pol = Poland_fig1.add_subplot(2, 2, 4)

ax_1_Pol.plot(years_Poland,Poland["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Pol.plot(years_Poland,Poland["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Pol.plot(years_Poland,Poland["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Pol.plot(years_Poland,Poland["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Pol.legend()
ax_2_Pol.legend()
ax_3_Pol.legend()
ax_4_Pol.legend()
plt.show()

Poland_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Poland")
plt.grid(False) 
plt.axis('off')
ax_5_Pol = Poland_fig2.add_subplot(2, 2, 1)
ax_6_Pol = Poland_fig2.add_subplot(2, 2, 2)
ax_7_Pol = Poland_fig2.add_subplot(2, 2, 3)
ax_8_Pol = Poland_fig2.add_subplot(2, 2, 4)

ax_5_Pol.plot(years_Poland,Poland["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Pol.plot(years_Poland,Poland["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Pol.plot(years_Poland,Poland["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Pol.plot(years_Poland,Poland["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Pol.legend()
ax_6_Pol.legend()
ax_7_Pol.legend()
ax_8_Pol.legend()
plt.show() 

Poland_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Pol = Poland_fig3.add_subplot(2, 1, 1)
ax_10_Pol = Poland_fig3.add_subplot(2, 1, 2)

ax_9_Pol.plot(years_Poland,Poland["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Pol.plot(years_Poland,Poland["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Pol.legend()
ax_10_Pol.legend()
plt.show()

# PORTUGAL

years_Portugal = Portugal["Year"]
Portugal_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Portugal")
plt.grid(False) 
plt.axis('off')

ax_1_Por = Portugal_fig1.add_subplot(2, 2, 1)
ax_2_Por = Portugal_fig1.add_subplot(2, 2, 2)
ax_3_Por = Portugal_fig1.add_subplot(2, 2, 3)
ax_4_Por = Portugal_fig1.add_subplot(2, 2, 4)

ax_1_Por.plot(years_Portugal,Portugal["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Por.plot(years_Portugal,Portugal["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Por.plot(years_Portugal,Portugal["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Por.plot(years_Portugal,Portugal["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Por.legend()
ax_2_Por.legend()
ax_3_Por.legend()
ax_4_Por.legend()
plt.show()

Portugal_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Portugal")
plt.grid(False) 
plt.axis('off')
ax_5_Por = Portugal_fig2.add_subplot(2, 2, 1)
ax_6_Por = Portugal_fig2.add_subplot(2, 2, 2)
ax_7_Por = Portugal_fig2.add_subplot(2, 2, 3)
ax_8_Por = Portugal_fig2.add_subplot(2, 2, 4)

ax_5_Por.plot(years_Portugal,Portugal["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Por.plot(years_Portugal,Portugal["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Por.plot(years_Portugal,Portugal["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Por.plot(years_Portugal,Portugal["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Por.legend()
ax_6_Por.legend()
ax_7_Por.legend()
ax_8_Por.legend()
plt.show() 

Portugal_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Por = Portugal_fig3.add_subplot(2, 1, 1)
ax_10_Por = Portugal_fig3.add_subplot(2, 1, 2)

ax_9_Por.plot(years_Portugal,Portugal["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Por.plot(years_Portugal,Portugal["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Por.legend()
ax_10_Por.legend()
plt.show()

# CZECH REPUBLIC

years_Czech = Czech_Republic["Year"]
Czech_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Czech Republic")
plt.grid(False) 
plt.axis('off')

ax_1_Cz = Czech_fig1.add_subplot(2, 2, 1)
ax_2_Cz = Czech_fig1.add_subplot(2, 2, 2)
ax_3_Cz = Czech_fig1.add_subplot(2, 2, 3)
ax_4_Cz = Czech_fig1.add_subplot(2, 2, 4)

ax_1_Cz.plot(years_Czech,Czech_Republic["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Cz.plot(years_Czech,Czech_Republic["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Cz.plot(years_Czech,Czech_Republic["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Cz.plot(years_Czech,Czech_Republic["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Cz.legend()
ax_2_Cz.legend()
ax_3_Cz.legend()
ax_4_Cz.legend()
plt.show()

Czech_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Czech Republic")
plt.grid(False) 
plt.axis('off')
ax_5_Cz = Czech_fig2.add_subplot(2, 2, 1)
ax_6_Cz = Czech_fig2.add_subplot(2, 2, 2)
ax_7_Cz = Czech_fig2.add_subplot(2, 2, 3)
ax_8_Cz = Czech_fig2.add_subplot(2, 2, 4)

ax_5_Cz.plot(years_Czech,Czech_Republic["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Cz.plot(years_Czech,Czech_Republic["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Cz.plot(years_Czech,Czech_Republic["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Cz.plot(years_Czech,Czech_Republic["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Cz.legend()
ax_6_Cz.legend()
ax_7_Cz.legend()
ax_8_Cz.legend()
plt.show() 

Czech_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Cz = Czech_fig3.add_subplot(2, 1, 1)
ax_10_Cz = Czech_fig3.add_subplot(2, 1, 2)

ax_9_Cz.plot(years_Czech,Czech_Republic["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Cz.plot(years_Czech,Czech_Republic["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Cz.legend()
ax_10_Cz.legend()
plt.show()

# ROMANIA

years_Romania = Romania["Year"]
Romania_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Romania")
plt.grid(False) 
plt.axis('off')

ax_1_Rom = Romania_fig1.add_subplot(2, 2, 1)
ax_2_Rom = Romania_fig1.add_subplot(2, 2, 2)
ax_3_Rom = Romania_fig1.add_subplot(2, 2, 3)
ax_4_Rom = Romania_fig1.add_subplot(2, 2, 4)

ax_1_Rom.plot(years_Romania,Romania["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Rom.plot(years_Romania,Romania["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Rom.plot(years_Romania,Romania["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Rom.plot(years_Romania,Romania["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Rom.legend()
ax_2_Rom.legend()
ax_3_Rom.legend()
ax_4_Rom.legend()
plt.show()

Romania_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Romania")
plt.grid(False) 
plt.axis('off')
ax_5_Rom = Romania_fig2.add_subplot(2, 2, 1)
ax_6_Rom = Romania_fig2.add_subplot(2, 2, 2)
ax_7_Rom = Romania_fig2.add_subplot(2, 2, 3)
ax_8_Rom = Romania_fig2.add_subplot(2, 2, 4)

ax_5_Rom.plot(years_Romania,Romania["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Rom.plot(years_Romania,Romania["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Rom.plot(years_Romania,Romania["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Rom.plot(years_Romania,Romania["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Rom.legend()
ax_6_Rom.legend()
ax_7_Rom.legend()
ax_8_Rom.legend()
plt.show() 

Romania_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Rom = Romania_fig3.add_subplot(2, 1, 1)
ax_10_Rom = Romania_fig3.add_subplot(2, 1, 2)

ax_9_Rom.plot(years_Romania,Romania["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Rom.plot(years_Romania,Romania["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Rom.legend()
ax_10_Rom.legend()
plt.show()

# SLOVENIA

years_Slovenia = Slovenia["Year"]
Slovenia_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Slovenia")
plt.grid(False) 
plt.axis('off')

ax_1_Slo = Slovenia_fig1.add_subplot(2, 2, 1)
ax_2_Slo = Slovenia_fig1.add_subplot(2, 2, 2)
ax_3_Slo = Slovenia_fig1.add_subplot(2, 2, 3)
ax_4_Slo = Slovenia_fig1.add_subplot(2, 2, 4)

ax_1_Slo.plot(years_Slovenia,Slovenia["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Slo.plot(years_Slovenia,Slovenia["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Slo.plot(years_Slovenia,Slovenia["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Slo.plot(years_Slovenia,Slovenia["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Slo.legend()
ax_2_Slo.legend()
ax_3_Slo.legend()
ax_4_Slo.legend()
plt.show()

Slovenia_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Slovenia")
plt.grid(False) 
plt.axis('off')
ax_5_Slo = Slovenia_fig2.add_subplot(2, 2, 1)
ax_6_Slo = Slovenia_fig2.add_subplot(2, 2, 2)
ax_7_Slo = Slovenia_fig2.add_subplot(2, 2, 3)
ax_8_Slo = Slovenia_fig2.add_subplot(2, 2, 4)

ax_5_Slo.plot(years_Slovenia,Slovenia["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Slo.plot(years_Slovenia,Slovenia["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Slo.plot(years_Slovenia,Slovenia["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Slo.plot(years_Slovenia,Slovenia["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Slo.legend()
ax_6_Slo.legend()
ax_7_Slo.legend()
ax_8_Slo.legend()
plt.show() 

Slovenia_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Slo = Slovenia_fig3.add_subplot(2, 1, 1)
ax_10_Slo = Slovenia_fig3.add_subplot(2, 1, 2)

ax_9_Slo.plot(years_Slovenia,Slovenia["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Slo.plot(years_Slovenia,Slovenia["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Slo.legend()
ax_10_Slo.legend()
plt.show()

# SWEDEN

years_Sweden = Sweden["Year"]
Sweden_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Sweden")
plt.grid(False) 
plt.axis('off')

ax_1_Swe = Sweden_fig1.add_subplot(2, 2, 1)
ax_2_Swe = Sweden_fig1.add_subplot(2, 2, 2)
ax_3_Swe = Sweden_fig1.add_subplot(2, 2, 3)
ax_4_Swe = Sweden_fig1.add_subplot(2, 2, 4)

ax_1_Swe.plot(years_Sweden, Sweden["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Swe.plot(years_Sweden, Sweden["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Swe.plot(years_Sweden, Sweden["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Swe.plot(years_Sweden, Sweden["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Swe.legend()
ax_2_Swe.legend()
ax_3_Swe.legend()
ax_4_Swe.legend()
plt.show()

Sweden_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Sweden")
plt.grid(False) 
plt.axis('off')
ax_5_Swe = Sweden_fig2.add_subplot(2, 2, 1)
ax_6_Swe = Sweden_fig2.add_subplot(2, 2, 2)
ax_7_Swe = Sweden_fig2.add_subplot(2, 2, 3)
ax_8_Swe = Sweden_fig2.add_subplot(2, 2, 4)

ax_5_Swe.plot(years_Sweden, Sweden["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_Swe.plot(years_Sweden, Sweden["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_Swe.plot(years_Sweden, Sweden["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_Swe.plot(years_Sweden, Sweden["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_Swe.legend()
ax_6_Swe.legend()
ax_7_Swe.legend()
ax_8_Swe.legend()
plt.show() 

Sweden_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_Swe = Sweden_fig3.add_subplot(2, 1, 1)
ax_10_Swe = Sweden_fig3.add_subplot(2, 1, 2)

ax_9_Swe.plot(years_Sweden, Sweden["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_Swe.plot(years_Sweden, Sweden["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_Swe.legend()
ax_10_Swe.legend()
plt.show()


# Now that I've plotted all the disorders for each European Country, I want to do a mean between all the States in order to see the European disorders trends and check ife there is any type of correlation.

Europe = pd.concat([Austria, Belgium, Bulgaria, Cyprus, Croatia, Denmark, Estonia, Finland, France, Germany, Greece, Slovakia, Spain, Hungary, Ireland, Italy, Latvia, Lithuania, Luxembourg, Malta, Netherlands, Poland, Portugal, Czech_Republic, Romania, Slovenia, Sweden], axis = 0)

# groupby year: I want to create a dataset with the mean values of all the countries
Europe_df_ = Europe.groupby("Year")
Europe_df = Europe_df_.mean()
Europe_df.info()
Europe_df.head()

# now I want to plot the Europe values

EU_fig1 = plt.figure(figsize=(20,20))
plt.title("Disorders across Europe")
plt.grid(False) 
plt.axis('off')

ax_1_Eu = EU_fig1.add_subplot(2, 2, 1)
ax_2_Eu = EU_fig1.add_subplot(2, 2, 2)
ax_3_Eu = EU_fig1.add_subplot(2, 2, 3)
ax_4_Eu = EU_fig1.add_subplot(2, 2, 4)

ax_1_Eu.plot(Europe_df.index, Europe_df["Schizophrenia (%)"],label='Schizophrenia (%)', color='red', marker = ".")
ax_2_Eu.plot(Europe_df.index, Europe_df["Bipolar disorder (%)"], label='Bipolar disorder (%)', color="blue", marker = ".")
ax_3_Eu.plot(Europe_df.index, Europe_df["Eating disorders (%)"], label='Eating disorders (%)', color="green", marker = ".")
ax_4_Eu.plot(Europe_df.index, Europe_df["Anxiety disorders (%)"], label='Anxiety disorders (%) ', color="orange", marker = ".")

ax_1_Eu.legend()
ax_2_Eu.legend()
ax_3_Eu.legend()
ax_4_Eu.legend()
plt.show()

EU_fig2 = plt.figure(figsize=(20,20))
plt.title("Disorders across Europe")
plt.grid(False) 
plt.axis('off')
ax_5_EU = EU_fig2.add_subplot(2, 2, 1)
ax_6_EU = EU_fig2.add_subplot(2, 2, 2)
ax_7_EU = EU_fig2.add_subplot(2, 2, 3)
ax_8_EU = EU_fig2.add_subplot(2, 2, 4)

ax_5_EU.plot(Europe_df.index, Europe_df["Drug use disorders (%)"], label='Drug use disorders (%)', color="red", marker = ".")
ax_6_EU.plot(Europe_df.index, Europe_df["Depression (%)"], label='Depression (%)', color="blue", marker = ".")
ax_7_EU.plot(Europe_df.index, Europe_df["Alcohol use disorders (%)"], label='Alcohol use disorders (%)', color="green", marker = ".")
ax_8_EU.plot(Europe_df.index, Europe_df["Suicide Rates"], label='Suicide Rates (over 100.000 deaths)', color="orange", marker = ".")

ax_5_EU.legend()
ax_6_EU.legend()
ax_7_EU.legend()
ax_8_EU.legend()
plt.show() 

EU_fig3 = plt.figure(figsize=(10,10))
plt.title("Prevalence in gender")
plt.grid(False) 
plt.axis('off')
ax_9_EU = EU_fig3.add_subplot(2, 1, 1)
ax_10_EU = EU_fig3.add_subplot(2, 1, 2)

ax_9_EU.plot(Europe_df.index, Europe_df["Prevalence in males (%)"], label='Prevalence in males (%)', color="red", marker = ".")
ax_10_EU.plot(Europe_df.index, Europe_df["Prevalence in females (%)"], label='Prevalence in females (%)', color="red", marker = ".")

ax_9_EU.legend()
ax_10_EU.legend()
plt.show()

