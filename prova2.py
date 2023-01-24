import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import streamlit as st
import seaborn as sb
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as km
#import tapby
mental_health = pd.read_csv("Mental_Dataset.csv") 

st.title("European Trends in Mental Health Project")
st.header("Aim of the project")
st.markdown("I worked on a dataset which offers a global vision of mental health disorders between 1990 and 2017 and I decided to focus my attention on the European situation. This project recurs to understand correlation between mental diseases inside each European Country. At the end of the presentation I would like to present, thanks to a cluster modelling, the similarities between the States mapping them geographically.")
st.header("Exploring and cleaning the dataset")
st.markdown("Firstly, I began by looking at the dataset information and I saw that there was many null values and many different type, mainly regarded as object. I started by changing  theses values into floats. Soon I realized that there is a problem in the Schizophrenia (%) column: apparently there was a string in the column at row 6468.  Working on data, I  discovered that the dataset offered by Kaggle was actually made by four different datasets merged vertically, instead of horizontally. Because of that, I had to understand where each dataset begins and ends, to separate them and to merge them horizontally. To create my final dataset, I decided to use only the first three “sub-dataset” since the fourth was ambiguous.")

index_to_keep_0 = np.arange(6468)
mental_1 = pd.read_csv("Mental_Dataset.csv").loc[index_to_keep_0]
index_to_keep = np.arange(6469, 54276)
mental_2 = pd.read_csv("Mental_Dataset.csv").loc[index_to_keep]

# MENTAL_1
mental_1 = mental_1.drop(["Code", "index"], axis = 1)
mental_1["Bipolar disorder (%)"] = mental_1["Bipolar disorder (%)"].astype(float) 
mental_1["Schizophrenia (%)"] = mental_1["Schizophrenia (%)"].astype(float)
mental_1["Eating disorders (%)"] = mental_1["Eating disorders (%)"].astype(float)  
mental_1["Year"] = mental_1["Year"].astype(float)
mental_1.info() # my data are clean now!

#MENTAL_2
mental_2 = mental_2.drop(["Alcohol use disorders (%)","Depression (%)", "Drug use disorders (%)", "Anxiety disorders (%)", "Code", "index"], axis = 1)
mental_2.rename(columns={"Schizophrenia (%)": "Prevalence in males", "Bipolar disorder (%)": "Prevalence in females", "Eating disorders (%)": "Population"}, inplace = True)
mental_2 = mental_2.dropna()
mental_2["Year"] = mental_2["Year"].astype(int)
mental_2["Prevalence in males"] = mental_2["Prevalence in males"].astype(float)
mental_2["Prevalence in females"] = mental_2["Prevalence in females"].astype(float)
mental_2["Population"] = mental_2["Population"].astype(float)
mental_2.info() #my data are clean now!

#MENTAL_3
true_index = np.arange(54277,102084)
mental_3 = pd.read_csv("Mental_Dataset.csv").loc[true_index]
mental_3 = mental_3.drop(["Alcohol use disorders (%)","Depression (%)", "Drug use disorders (%)", "Anxiety disorders (%)", "Entity", "Year"], axis = 1)
mental_3.rename(columns={"Schizophrenia (%)": "Suicide Rates", "Bipolar disorder (%)": "Depressive Disorder Rates", "Eating disorders (%)": "Population"}, inplace = True)
mental_3 = mental_3.drop(["Code", "index"], axis = 1) 
mental_3 = mental_3.dropna() 
mental_3["Suicide Rates"] = mental_3["Suicide Rates"].astype(float)
mental_3["Depressive Disorder Rates"] = mental_3["Depressive Disorder Rates"].astype(float)
mental_3["Population"] = mental_3["Population"].astype(float)
mental_3.info() #my data are clean now!

#FINAL DATASET
mental_health_ = pd.merge(mental_1,mental_2)
mental_health_['Year'] = mental_health_['Year'].astype(int)

mental_health_.index = mental_3.index
mental_health_final = pd.concat([mental_health_, mental_3], axis=1)
mental_health_final = mental_health_final.drop(["Population"], axis=1)
mental_health_final.rename(columns={'Entity': 'Country'}, inplace=True)
mental_health_final = mental_health_final.drop(["Depressive Disorder Rates"], axis=1)
mental_health_final["Suicide Rates"] = (mental_health_final["Suicide Rates"]/100000*100)

mental_health_final.info() # My final dataset is clean!
mental_health_final.to_csv('mental_health_final.csv', index=False)

#PLOTS

st.markdown("In the following section of my presentation I show for each European country the trends in disorders, the prevalence in gender and the correlation between each disorder inside each country.")

Austria= mental_health_final.loc[mental_health_final["Country"] == "Austria"].set_index("Year")
Belgium = mental_health_final.loc[mental_health_final["Country"] == "Belgium"].set_index("Year")
Bulgaria= mental_health_final.loc[mental_health_final["Country"] == "Bulgaria"].set_index("Year")
Cyprus = mental_health_final.loc[mental_health_final["Country"] == "Cyprus"].set_index("Year")
Croatia = mental_health_final.loc[mental_health_final["Country"] == "Croatia"].set_index("Year")
Denmark = mental_health_final.loc[mental_health_final["Country"] == "Denmark"].set_index("Year")
Estonia = mental_health_final.loc[mental_health_final["Country"] == "Estonia"].set_index("Year")
Finland = mental_health_final.loc[mental_health_final["Country"] == "Finland"].set_index("Year")
France = mental_health_final.loc[mental_health_final["Country"] == "France"].set_index("Year")
Germany = mental_health_final.loc[mental_health_final["Country"] == "Germany"].set_index("Year")
Greece = mental_health_final.loc[mental_health_final["Country"] == "Greece"].set_index("Year")
Slovakia = mental_health_final.loc[mental_health_final["Country"] == "Slovakia"].set_index("Year")
Spain = mental_health_final.loc[mental_health_final["Country"] == "Spain"].set_index("Year")
Hungary = mental_health_final.loc[mental_health_final["Country"] == "Hungary"].set_index("Year")
Ireland = mental_health_final.loc[mental_health_final["Country"] == "Ireland"].set_index("Year")
Italy = mental_health_final.loc[mental_health_final["Country"] == "Italy"].set_index("Year")
Latvia = mental_health_final.loc[mental_health_final["Country"] == "Latvia"].set_index("Year")
Lithuania = mental_health_final.loc[mental_health_final["Country"] == "Lithuania"].set_index("Year")
Luxembourg = mental_health_final.loc[mental_health_final["Country"] == "Luxembourg"].set_index("Year")
Malta = mental_health_final.loc[mental_health_final["Country"] == "Malta"].set_index("Year")
Netherlands = mental_health_final.loc[mental_health_final["Country"] == "Netherlands"].set_index("Year")
Poland = mental_health_final.loc[mental_health_final["Country"] == "Poland"].set_index("Year")
Portugal = mental_health_final.loc[mental_health_final["Country"] == "Portugal"].set_index("Year")
Czech_Republic= mental_health_final.loc[mental_health_final["Country"] == "Czech Republic"].set_index("Year")
Romania = mental_health_final.loc[mental_health_final["Country"] == "Romania"].set_index("Year")
Slovenia = mental_health_final.loc[mental_health_final["Country"] == "Slovenia"].set_index("Year")
Sweden = mental_health_final.loc[mental_health_final["Country"] == "Sweden"].set_index("Year")

# STREAMLIT SETTINGS
# sidebar
countries_dict = {"Austria":Austria, "Belgium":Belgium, "Bulgaria":Bulgaria, "Cyprus": Cyprus, "Croatia":Croatia, "Denmark":Denmark, "Estonia":Estonia, "Finland":Finland, "France":France, "Germany":Germany, "Greece":Greece, "Slovakia":Slovakia, "Spain":Spain, "Hungary":Hungary, "Ireland":Ireland, "Italy":Italy, "Latvia":Latvia, "Lithuania":Lithuania, "Luxembourg":Luxembourg, "Malta":Malta, "Netherlands":Netherlands, "Poland":Poland, "Portugal":Portugal, "Czech Republic":Czech_Republic, "Romania":Romania, "Slovenia":Slovenia,"Sweden": Sweden}
st.sidebar.title("Plots")
for country in countries_dict.keys():
    if st.sidebar.button(country):
        st.subheader(f"Disorders across {country} 1990-2017")

#PLOTS, SUBHEADERS, TABS AND EXAPNDERS
colors=['blue','red','green','purple','orange','brown','pink','gray']

# AUSTRIA
figAU, axsAU = plt.subplots(4, 2, figsize=(8,15))
axs = axsAU.ravel()
columns_to_plot = Austria.loc[:,~Austria.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Austria.index, Austria[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
    plt.tight_layout()

# PREVALENCE DISTRIBUTION
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Belgium.columns[8:10]):
    axs[i].plot(Austria.index, Austria[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
    plt.tight_layout()

st.subheader("Disorders across Austria 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figAU)
    st.write("As it can be seen from the plots, Schizophrenia and Eating disords grew significantly in the last years, whereas the other disorders decreased. Although the drug use disorder is still high, from the 2018 until 2017 it slowly decreased.")
with st.expander("Prevalence in distribution"):
    st.pyplot(fig)
    st.write("Fortunately, the number of both the females and males affected by any kind of disorders is strictly decreased.")
with st.expander("Correlation between disorders"):
    fig_corr = plt.figure(figsize=(6,5))
    sb.heatmap(Austria.corr(), annot=True)
    st.pyplot(fig_corr)
    st.write("The heatmap above shows the correlation between each disorders. It is also interesting to check how the gender is correlated to each disorder.")

#BELGIUM
figBE, axsBE = plt.subplots(4, 2, figsize=(8, 15))
axsBE = axsBE.ravel()
columns_to_plot = Belgium.loc[:,~Belgium.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axsBE[i].plot(Belgium.index,Belgium[col],'o-',color=colors[i])
    axsBE[i].set_title(col)
    axsBE[i].set_xlabel('years')
    axsBE[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Austria.columns[8:10]):
    axs[i].plot(Belgium.index,Belgium[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Belgium 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figBE)
    st.write("As it can be seen from the plots, Schizophrenia and Eating disords grew significantly in the last years, whereas the other disorders decreased. Although the drug use disorder is still high, from the 2018 until 2017 it slowly decreased.")
with st.expander("Prevalence in distribution"):
    st.pyplot(fig)
    st.write("Fortunately, the number of both the females and males affected by any kind of disorders is strictly decreased.")
with st.expander("Correlation between disorders"):
    fig_corr1= plt.figure(figsize=(8,6))
    sb.heatmap(Belgium.corr(), annot=True)
    st.pyplot(fig_corr1)

# BULGARIA
figBU, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Bulgaria.loc[:,~Bulgaria.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Bulgaria.index,Bulgaria[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figBU1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Bulgaria.columns[8:10]):
    axs[i].plot(Bulgaria.index,Bulgaria[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Bulgaria 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figBU)
    st.write("As we can see from the graph, there is a steady increase in many disorders, such as Schizophrenia, Bipolarism, Anxienty, Drug abuse and Eating disorders. Whereas for depression and suicide rates is the opposite situation. It is a slightly decrease in Alchool abuse from 2010.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figBU1)
    st.write("The prevalence in gender follows the same decreasing trend in both females and males.")
with st.expander("Correlation between disorders"):
    j = plt.figure(figsize=(8,6))
    sb.heatmap(Bulgaria.corr(), annot=True)
    st.pyplot(j)
    st.write()

#CYPRUS
figCY, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Cyprus.loc[:,~Cyprus.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Cyprus.index,Cyprus[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figCY1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Cyprus.columns[8:10]):
    axs[i].plot(Cyprus.index,Cyprus[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Cyprus 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figCY)
    st.write("As it can be seen from the plots, Schizophrenia and Eating disords grew significantly in the last years, whereas the other disorders decreased. Although the drug use disorder is still high, from the 2018 until 2017 it slowly decreased.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figCY1)
    st.write("Fortunately, the number of both the females and males affected by any kind of disorders is strictly decreased.")
with st.expander("Correlation between disorders"):
    a= plt.figure(figsize=(8,6))
    sb.heatmap(Cyprus.corr(), annot=True)
    st.pyplot(a)

#CROATIA
figCRO, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Croatia.loc[:,~Croatia.columns.isin(['Country', 'Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Croatia.index,Croatia[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figCRO1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Croatia.columns[8:10]):
    axs[i].plot(Croatia.index,Croatia[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Croatia 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figCRO)
    st.write("The plots show an uptrend in Schizofrenia, Bipolarism and in Eatind Disorders. On the other hand, the remaining ones are showing a decrese in numbers of cases. The alcohol abuse, after a sharp increase from the 2000, is now gradually declining.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figCRO1)
    st.write("Both females and males trends show a rapid decrease over the years. Now the percentage of people affected is almost constant.")
with st.expander("Correlation between disorders"):
    q = plt.figure(figsize=(8,6))
    sb.heatmap(Croatia.corr(), annot=True)
    st.pyplot(q)

# DENMARK
figDE, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Denmark.loc[:,~Denmark.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Denmark.index,Denmark[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figDE1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Denmark.columns[8:10]):
    axs[i].plot(Denmark.index,Denmark[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Denmark 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figDE)
    st.write("It is striking to see the steady increase in Schizophrenia and in Drug use disorders, respectively from 2005 and 2010. On the other hand, Bipolar Disorder has a marked drop from 2000. Depression and Suicide rates follow a similiar decreasing pattern, like Alcohol and Anxiety Disorders.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figDE1)
    st.write("Both females and males trends show a rapid decrease over the years. Now the percentage of people affected is almost constant.")
with st.expander("Correlation between disorders"):
    w =  plt.figure(figsize=(8,6))
    sb.heatmap(Denmark.corr(), annot=True)
    st.pyplot(w)

# ESTONIA
figES, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Estonia.loc[:,~Estonia.columns.isin(['Country', 'Year', 'Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Estonia.index,Estonia[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figES1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Estonia.columns[8:10]):
    axs[i].plot(Estonia.index,Estonia[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Estonia 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figES)
    st.write("Schizophrenia, Bipolarism, Drug use Disorder and Eating Disorder are following an increasing trend. On the contrary, Suicide Rates and Depression are decreasing. Anxiety disorders and Alcohol abuse have a bumpy trends but now they are bothe slightly decreasing. ")
with st.expander("Prevalence in distribution"):
    st.pyplot(figES1)
    st.write("Both females and males trends show a rapid decrease over the years. Now the percentage of people affected is almost constant.")
with st.expander("Correlation between disorders"):
    e = plt.figure(figsize=(8,6))
    sb.heatmap(Estonia.corr(), annot=True)
    st.pyplot(e)

# FINLAND
figFI, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Finland.loc[:,~Finland.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Finland.index,Finland[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figFI1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Finland.columns[8:10]):
    axs[i].plot(Finland.index, Finland[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Finland 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figFI)
    st.write("On the one hand, Depression and Suicide Rates are gradually decreasing over the years. On the other hand, Schizophrenia steeply decreases from 2010. Similarly it happens to Bipolar Disorder and Anxiety disorder, but they now have an increasing trends. Both Eating Disorders and Drug use Disorder have a rise in the number of cases.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figFI1)
    st.write("Both females and males trends show a slow decrease over the years.")
with st.expander("Correlation between disorders"):
    R = plt.figure(figsize=(8,6))
    sb.heatmap(Finland.corr(), annot=True)
    st.pyplot(R)

#FRANCE
figFR, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = France.loc[:,~France.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(France.index, France[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figFR1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(France.columns[8:10]):
    axs[i].plot(France.index,France[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across France 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figFR)
    st.write("Schizophrenia, Bipolarism and Anxiety Disorders follow a similar path, they increase from 1990 to 2000-2005 and then they constantly decrease. As we can see from the plots, Eating Disorder is negative correlated to Depression and Suicide Rates since they have opposite trends. Drug and Alcohol use Disorders also have a similar path and they both start falling in numbers around 2010.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figFR1)
    st.write("Both females and males trends show a decrease over the years, but around 2005 the number of people affected by disorders arises to then decrease again around 2010.")
with st.expander("Correlation between disorders"):
    t= plt.figure(figsize=(8,6))
    sb.heatmap(France.corr(), annot=True)
    st.pyplot(t)

# GERMANY
figGE, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Germany.loc[:,~Germany.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Germany.index,Germany[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figGE1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Germany.columns[8:10]):
    axs[i].plot(Germany.index, Germany[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Germany 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figGE)
    st.write("Over the years, the number of cases afflicted by Alcohol and Anxiety disorders has decreased as well as the Suicide Rates. On the other hand, the percentage of people affected by Depression, Bipolarism and Drug use Disorder has increased.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figGE1)
    st.write("The trend in prevalence is different for the gender. From 2005, the prevalence in males starts decreasing, whereas the prevalence in females rises even more after a steep increase that began in 2000..")
with st.expander("Correlation between disorders"):
    u = plt.figure(figsize=(8,6))
    sb.heatmap(Germany.corr(), annot=True)
    st.pyplot(u)

# GREECE
figGRE, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Greece.loc[:,~Greece.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Greece.index,Greece[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figGRE1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Greece.columns[8:10]):
    axs[i].plot(Greece.index,Greece[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Greece 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figGRE)
    st.write("Schizophrenia and Anxiety disorders follow a similar path by increasing from 2000 and slightly decreasing by 2010. Similar happen to Eating and Alcohol use disorders. As can be seen from the plots, many changes in trends happen around 2000 and 2005. The most visible change is in Suicide rates: after reaching the lowest point in 2002, it can be clearly seen the phenomenal growth until 2015.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figGRE1)
    st.write("The prevalence in gender follow the same trends: the percentage of people affected by disorders increases until 2005 when the rates start falling.")
with st.expander("Correlation between disorders"):
    I=plt.figure(figsize=(8,6))
    sb.heatmap(Greece.corr(), annot=True)
    st.pyplot(I)

# SLOVAKIA
figSLO, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Slovakia.loc[:,~Slovakia.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Slovakia.index,Slovakia[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figSLO1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Slovakia.columns[8:10]):
    axs[i].plot(Slovakia.index,Slovakia[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Slovakia 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figSLO)
    st.write("Schizophrenia and Anxiety disorders follow a similar path by increasing from 2000 and slightly decreasing by 2010. Similar happen to Eating and Alcohol use disorders. As can be seen from the plots, many changes in trends happen around 2000 and 2005. The most visible change is in Suicide rates: after reaching the lowest point in 2002, it can be clearly seen the phenomenal growth until 2015.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figSLO1)
    st.write("The prevalence in gender follow the same trends: the percentage of people affected by disorders increases until 2005 when the rates start falling.")
with st.expander("Correlation between disorders"):
    o=plt.figure(figsize=(8,6))
    sb.heatmap(Slovakia.corr(), annot=True)
    st.pyplot(o)

# SPAIN
figSPA, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Spain.loc[:,~Spain.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Spain.index,Spain[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figSPA1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Spain.columns[8:10]):
    axs[i].plot(Spain.index,Spain[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Spain 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figSPA)
    st.write("Overall, it stands out that most of the charts are showing a general increase in rates. Bipolar Disorder steady rises from 2000, as well as Anxiety Disorder. Depression and Suicide rates have a decreasing path, even though the first changes its trend from 2005.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figSPA1)
    st.write("Both females and males trends fall over the years but from 2005 the percentage of people affected rises.")
with st.expander("Correlation between disorders"):
    P=plt.figure(figsize=(8,6))
    sb.heatmap(Spain.corr(), annot=True)
    st.pyplot(P)

# HUNGARY
figHU, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Hungary.loc[:,~Hungary.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Hungary.index,Hungary[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figHU1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Hungary.columns[8:10]):
    axs[i].plot(Hungary.index,Hungary[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Hungary 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figHU)
    st.write("Similar to the previous Country, most of the charts are showing a general increase in rates. Only two charts are depicting a decreasing path: Suicide Rates and Depression")
with st.expander("Prevalence in distribution"):
    st.pyplot(figHU1)
    st.write("Both females and males have a decreasing trends.")
with st.expander("Correlation between disorders"):
    A= plt.figure(figsize=(8,6))
    sb.heatmap(Hungary.corr(), annot=True)
    st.pyplot(A)

#IRELAND
figIR, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Ireland.loc[:,~Ireland.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Ireland.index,Ireland[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figIR1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Ireland.columns[8:10]):
    axs[i].plot(Ireland.index,Ireland[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Ireland 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figIR)
    st.write("It is striking that, overall, the plots follow an identical continual growth over the years. Depression and Drug use Disorder also follow this trend but from 2005 the percentage start falling. In parallel, Suicide Rates has dropped since 2000.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figIR1)
    st.write("Both females and males have an increasing trends until 2005 when they both start droppong.")
    F= plt.figure(figsize=(8,6))
    sb.heatmap(Ireland.corr(), annot=True)
    st.pyplot(F)

#ITALY
figIT, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Italy.loc[:,~Italy.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Italy.index,Italy[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
# PREVALENCE DISTRIBUTION
figIT1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Italy.columns[8:10]):
    axs[i].plot(Italy.index,Italy[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Italy 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figIT)
    st.write("It can clearly be seen that there is a slight rise in Eating Disorder, Depression and Bipolarism in the last 10 years. Depression and Suicide Rates follow a similar decreasing trend. Both Anxiety and Alcohol Abuse rise from 2000, reaching the maximum level around 2010.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figIT1)
    st.write("Both females and males have a decreasing trends until 2005 when it start slightly increasing.")
with st.expander("Correlation between disorders"):
    S= plt.figure(figsize=(8,6))
    sb.heatmap(Italy.corr(), annot=True)
    st.pyplot(S)

# LATVIA
figLAT, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Latvia.loc[:,~Latvia.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Latvia.index,Latvia[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figLAT1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Latvia.columns[8:10]):
    axs[i].plot(Latvia.index,Latvia[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Latvia 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figLAT)
    st.write("Many of the disorders have a decreasing trend until 2000 when it starts an opposite increasing path. Alcohol Abuse, Suicide Rates and Depression follow a falling path.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figLAT1)
    st.write("Both females and males have a decreasing trend that is constant from 2010.")
with st.expander("Correlation between disorders"):
    F= plt.figure(figsize=(8,6))
    sb.heatmap(Latvia.corr(), annot=True)
    st.pyplot(F)

# LITHUANIA
figLIT, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Lithuania.loc[:,~Lithuania.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Lithuania.index,Lithuania[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figLIT1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Lithuania.columns[8:10]):
    axs[i].plot(Lithuania.index,Lithuania[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Lithuania 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figLIT)
    st.write("Many of the disorders start to have an increasing trend from 2000-2005. On the other hand, Alcohol Abuse, Suicide Rates and Depression have an opposite behaviour.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figLIT1)
    st.write("Both females and males have an increasing trend until 2000 when it starts falling.")
with st.expander("Correlation between disorders"):
    h=plt.figure(figsize=(8,6))
    sb.heatmap(Lithuania.corr(), annot=True)
    st.pyplot(h)

# LUXEMBOURG
figLUX, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Luxembourg.loc[:,~Luxembourg.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Luxembourg.index,Luxembourg[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
# PREVALENCE DISTRIBUTION
figLUX1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Luxembourg.columns[8:10]):
    axs[i].plot(Luxembourg.index,Luxembourg[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Luxembourg 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figLIT)
    st.write("The majority of the plots show a decreasing trend that starts between the last year of the nineties and the beginning of the XXI century. On the other hand, Schizophrenia and Eating Disorder have an opposite behaviour being positively correlated to each other.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figLIT1)
    st.write("Both females and males have a decreasing trend which is now constant.")
with st.expander("Correlation between disorders"):
    J=plt.figure(figsize=(8,6))
    sb.heatmap(Malta.corr(), annot=True)
    st.pyplot(J)

# HOLLAND
figHO, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Netherlands.loc[:,~Netherlands.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Netherlands.index,Netherlands[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figHO1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Netherlands.columns[8:10]):
    axs[i].plot(Netherlands.index,Netherlands[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Netherlands 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figLIT)
    st.write("Anxienty Disorder, Alcohol Abuse and Depression follow a similar trend. Overall, the percentage of people affected by mental disorders is decreased over time, a part for Alcholism, Bipolarism and Eating Disorders.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figLIT1)
    st.write("The trends are similar, however, the percentage of males affected is higher than the females.")
with st.expander("Correlation between disorders"):
    L= plt.figure(figsize=(8,6))
    sb.heatmap(Netherlands.corr(), annot=True)
    st.pyplot(L)

# POLAND
figPO, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Poland.loc[:,~Poland.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Poland.index,Poland[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
# PREVALENCE DISTRIBUTION
figPO1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Poland.columns[8:10]):
    axs[i].plot(Poland.index,Poland[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Poland 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figPO)
    st.write("The majority of the plots show an increase in percentage of people suffering of mental health disorder. Howeevr, the number of people affected by Depression, Drug addiction is falling, as well as Suicide Rates.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figPO1)
    st.write("The plots show that the number of males affected by mental disorders is close to the females, even if the latter has decreased from 2000.")
with st.expander("Correlation between disorders"):
    Z=plt.figure(figsize=(8,6))
    sb.heatmap(Poland.corr(), annot=True)
    st.pyplot(Z)

# PORTUGAL
figPOR, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Portugal.loc[:,~Portugal.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Portugal.index,Portugal[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figPOR1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Portugal.columns[8:10]):
    axs[i].plot(Portugal.index,Portugal[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Portugal 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figPOR)
    st.write("Schizophrenia, Anxiety and Eating Disorders have a similar costant increasing trend. Drug use disorder and Bipolarism also have a similar bumpy path, reaching the lowest level in 2010 and 2005 respectively.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figPOR1)
    st.write("Prevalence in females and males have a common path where they reach the highest level in 2005 which is followed by a steady decrease.")
with st.expander("Correlation between disorders"):
    v=plt.figure(figsize=(8,6))
    sb.heatmap(Portugal.corr(), annot=True)
    st.pyplot(v)

# CZECH REPUBLIC
figCZ, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Czech_Republic.loc[:,~Czech_Republic.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Czech_Republic.index,Czech_Republic[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figCZ1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Czech_Republic.columns[8:10]):
    axs[i].plot(Czech_Republic.index,Czech_Republic[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Czech Republic 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figCZ)
    st.write("Overall, the percentage of people affected by mental health disorders increases over time. However, Depression, Suicide Rates and Anxiety Disorder decreases over the years. The latter, in particular, reach the highest level around 2005 when it starts a decreasing trend.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figCZ1)
    st.write("The trends are similar, however, the percentage of females affected is higher than the males.")
with st.expander("Correlation between disorders"):
    m=plt.figure(figsize=(8,6))
    sb.heatmap(Czech_Republic.corr(), annot=True)
    st.pyplot(m)

# ROMANIA
figRO, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Romania.loc[:,~Romania.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Romania.index,Romania[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figRO1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Romania.columns[8:10]):
    axs[i].plot(Romania.index,Romania[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Romania 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figRO)
    st.write("Schizophrenia, Eating Disorder and Bipolarism have a similar increasing path, which is the opposite of Alcoholism. After having reached the minimum respectively in 2001 and 2010, Drug Addiction and Depression have a steady increase.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figRO1)
    st.write("The trends in gender prevalence are exactly the opposite: on th one hand, the males affected by disorders increases over time, on the other hand, the percentage of females decreases.")
with st.expander("Correlation between disorders"):
    N=plt.figure(figsize=(8,6))
    sb.heatmap(Romania.corr(), annot=True)
    st.pyplot(N)

# SLOVENIA
figSLOV, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Slovenia.loc[:,~Slovenia.columns.isin(['Country', 'Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Slovenia.index,Slovenia[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
# PREVALENCE DISTRIBUTION
figSLOV1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Slovenia.columns[8:10]):
    axs[i].plot(Slovenia.index,Slovenia[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Slovenia 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figSLOV)
    st.write("Schizophrenia, Bipolarism, Eating, Drug and Alcohol use disorders increases over time, whereas the other disorders decreases.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figSLOV1)
    st.write("The trends are similar, however, the percentage of females affected is higher than the males.")
with st.expander("Correlation between disorders"):
    f=plt.figure(figsize=(8,6))
    sb.heatmap(Slovenia.corr(), annot=True)
    st.pyplot(f)

# SWEDEN
figSWE, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
columns_to_plot = Sweden.loc[:,~Sweden.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]
for i, col in enumerate(columns_to_plot):
    axs[i].plot(Sweden.index,Sweden[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

# PREVALENCE DISTRIBUTION
figSWE1, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Sweden.columns[8:10]):
    axs[i].plot(Sweden.index,Sweden[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()

st.subheader("Disorders across Sweden 1990-2017")
with st.expander("Plots of the disorders"):
    st.pyplot(figSWE)
    st.write("Bipolarism and Anxiety disorder follow almost the same trend, where the number of cases decreases from 2005. Also Schizophrenia, Depressin and Suicide rates decreases over time. However, the number of people suffering by Eating and Drug use disorder increases.")
with st.expander("Prevalence in distribution"):
    st.pyplot(figSWE1)
    st.write("The prevalence in gender follow in both cases a decreasing path with a slight increase between 2000 and 2005 for the males and between 2005 and 2010 for the females.")
with st.expander("Correlation between disorders"):
    z=plt.figure(figsize=(8,6))
    sb.heatmap(Sweden.corr(), annot=True)
    st.pyplot(z)

#EUROPE and WORLD
Europe = pd.concat([Austria, Belgium, Bulgaria, Cyprus, Croatia, Denmark, Estonia, Finland, France, Germany, Greece, Slovakia, Spain, Hungary, Ireland, Italy, Latvia, Lithuania, Luxembourg, Malta, Netherlands, Poland, Portugal, Czech_Republic, Romania, Slovenia, Sweden], axis = 0)
Europe_df_ = Europe.groupby("Year")
Europe_df = Europe_df_.mean()

World_df_ = mental_health_final.groupby("Year")
World_df = World_df_.mean()

# CLUSTERING EU
x = [Europe_df["Schizophrenia (%)"], Europe_df["Bipolar disorder (%)"], Europe_df["Eating disorders (%)"], Europe_df["Anxiety disorders (%)"], Europe_df["Drug use disorders (%)"], Europe_df["Depression (%)"], Europe_df["Alcohol use disorders (%)"], Europe_df["Prevalence in males"], Europe_df["Prevalence in females"], Europe_df["Suicide Rates"]]
y = Europe_df.index

st.title("Cluster analysis: KMeans Algorithm on the European situation")
st.write("After having created a dataset with all the 27 European Countries, I decided to conduct a clustering analysis to find patterns in the data by grouping similar data points together. On the x-axis there are the mental disorders, while on the y-axis the European countries. Thanks to the KMeans method I was able to check the similarities in disorders across countries.")

x, y = make_blobs(n_samples=200, n_features=2, centers= 5, cluster_std=0.8, random_state=42)
km_eu = km(n_clusters=5, init="random", n_init=10, max_iter=100, tol = 1e-04, random_state=0)
y_km= km_eu.fit_predict(x)
EU_scatter= plt.figure(figsize=(10, 8))
for i in range(5):
  plt.scatter(x[y_km == i, 0], x[y_km ==i, 1])
plt.legend(["cluster 1", "cluster 2", "cluster 3", "cluster 4", "cluster 5"])
st.pyplot(EU_scatter)

# CLUSTERS COUNTRIES EU
clusters = km_eu.fit_predict(Europe_df)
Europe_df["cluster"] = clusters

one_df = Europe_df.loc[Europe_df["cluster"]==0]
two_df = Europe_df.loc[Europe_df["cluster"]==1]
three_df = Europe_df.loc[Europe_df["cluster"]==2]
four_df = Europe_df.loc[Europe_df["cluster"]==3]
five_df = Europe_df.loc[Europe_df["cluster"]==4]

st.write("From the scatter plot above it can be seen that I clustered the Countries into 5 groups. In  the first cluster there are Portugal, Sweden and Finland; in the second one, Austria, Belgium, Cyprus, Denmark, Italy, Luxembourg, Malta and Spain. The third cluster is made by France, Germany, Greece, Ireland and Netherlands, while the fourth cluster by Bulgaria, Croatia, Czech Republic, Hungary, Poland, Romania, Slovakia and Slovenia. In the last cluster there are Estonia, Latvia and Lithuania.")