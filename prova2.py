import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import streamlit as st
import seaborn as sb
#import tapby
mental_health = pd.read_csv("Mental_Dataset.csv") 

st.title("European Trends in Mental Health Project")
st.header("Aim of the project")
st.markdown("This project aims to study the mental disorders trends across Europe trying to understand possible correlations and future trends. I have worked on dataset which reports many mental disors around the world but I decided to focus on the European situation.\n Since the data provided go from 1990 to 2017, I also wanted to check if the analyzed future trends correspod with the present situation. ")

st.header("Exploring and cleaning the dataset")
st.markdown("The first thing I did is checking the information of the dataset and I see that there were many null values and numerical values that were indicated as object. Because of that, firstly I wanted to change the object values into floats, starting by the Schizophrenia (%) column. Soon I realized that there is a problem in the column Schizophrenia (%). Apparentely there was a string in the column at row 6468.")
st.markdown(" I discovered a very interesting thing: there was the possibility that my dataframe is actually made by more datasets put together. To check that the row 6468 contains the keys of a whole new dataframe, I print the following five rows just to be sure of my supposition.")
st.markdown("As I thought, this is the beginning of a whole new dataset. I have to split mental_health into parts and merge them horizontally (probably the owner has concat them vertically).")

index_to_keep_0 = np.arange(6468)
mental_1 = pd.read_csv("Mental_Dataset.csv").loc[index_to_keep_0]
index_to_keep = np.arange(6469, 54276)
mental_2 = pd.read_csv("Mental_Dataset.csv").loc[index_to_keep]

# MENTAL_1
mental_1.info()
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
mental_health_.head().T
mental_health_['Year'] = mental_health_['Year'].astype(int)
mental_health_.info()

mental_health_.index = mental_3.index
mental_health_final = pd.concat([mental_health_, mental_3], axis=1)
mental_health_final = mental_health_final.drop(["Population"], axis=1)
mental_health_final.rename(columns={'Entity': 'Country'}, inplace=True)
mental_health_final = mental_health_final.drop(["Depressive Disorder Rates"], axis=1)
mental_health_final["Suicide Rates"] = (mental_health_final["Suicide Rates"]/100000*100)

mental_health_final.info() # My final dataset is clean!
mental_health_final.to_csv('mental_health_final.csv', index=False)

#PLOTS

st.markdown("At this point of the project I decided to analyze the situation in each European country plotting the trends of the disorders")

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
st.subheader("Disorders across Cyprus 1990-2017")
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
st.subheader("Disorders across Croatia 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Croatia.loc[:,~Croatia.columns.isin(['Country', 'Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Croatia.index,Croatia[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Croatia.columns[8:10]):
    axs[i].plot(Croatia.index,Croatia[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
q = plt.figure(figsize=(8,6))
sb.heatmap(Croatia.corr(), annot=True)
plt.show()
st.pyplot(q)

# DENMARK

st.subheader("Disorders across Denmark 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Denmark.loc[:,~Denmark.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Denmark.index,Denmark[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Denmark.columns[8:10]):
    axs[i].plot(Denmark.index,Denmark[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
w =  plt.figure(figsize=(8,6))
sb.heatmap(Denmark.corr(), annot=True)
st.pyplot(w)

# ESTONIA
st.subheader("Disorders across Estonia 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Estonia.loc[:,~Estonia.columns.isin(['Country', 'Year', 'Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Estonia.index,Estonia[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Estonia.columns[8:10]):
    axs[i].plot(Estonia.index,Estonia[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
e = plt.figure(figsize=(8,6))
sb.heatmap(Estonia.corr(), annot=True)
st.pyplot(e)

# FINLAND
st.subheader("Disorders across Finland 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Finland.loc[:,~Finland.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Finland.index,Finland[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Finland.columns[8:10]):
    axs[i].plot(Finland.index, Finland[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
R = plt.figure(figsize=(8,6))
sb.heatmap(Finland.corr(), annot=True)
st.pyplot(R)

# FRANCE
st.subheader("Disorders across France 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = France.loc[:,~France.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(France.index, France[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(France.columns[8:10]):
    axs[i].plot(France.index,France[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
t= plt.figure(figsize=(8,6))
sb.heatmap(France.corr(), annot=True)
st.pyplot(t)

# GERMANY
st.subheader("Disorders across Germany 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Germany.loc[:,~Germany.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Germany.index,Germany[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Germany.columns[8:10]):
    axs[i].plot(Germany.index, Germany[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
u = plt.figure(figsize=(8,6))
sb.heatmap(Germany.corr(), annot=True)
st.pyplot(u)

# GREECE
st.subheader("Disorders across Greece 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Greece.loc[:,~Greece.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Greece.index,Greece[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Greece.columns[9:11]):
    axs[i].plot(Greece.index,Greece[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
I=plt.figure(figsize=(8,6))
sb.heatmap(Greece.corr(), annot=True)
st.pyplot(I)

# SLOVAKIA
st.subheader("Disorders across Slovakia 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Slovakia.loc[:,~Slovakia.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Slovakia.index,Slovakia[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Slovakia.columns[8:10]):
    axs[i].plot(Slovakia.index,Slovakia[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
o=plt.figure(figsize=(8,6))
sb.heatmap(Slovakia.corr(), annot=True)
st.pyplot(o)

# SPAIN
st.subheader("Disorders across Spain 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Spain.loc[:,~Spain.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Spain.index,Spain[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Spain.columns[8:10]):
    axs[i].plot(Spain.index,Spain[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
P=plt.figure(figsize=(8,6))
sb.heatmap(Spain.corr(), annot=True)
st.pyplot(P)

# HUNGARY
st.subheader("Disorders across Hungary 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Hungary.loc[:,~Hungary.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Hungary.index,Hungary[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Hungary.columns[8:10]):
    axs[i].plot(Hungary.index,Hungary[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
A= plt.figure(figsize=(8,6))
sb.heatmap(Hungary.corr(), annot=True)
st.pyplot(A)

# IRELAND
st.subheader("Disorders across Ireland 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Ireland.loc[:,~Ireland.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Ireland.index,Ireland[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Ireland.columns[8:10]):
    axs[i].plot(Ireland.index,Ireland[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
F= plt.figure(figsize=(8,6))
sb.heatmap(Ireland.corr(), annot=True)
st.pyplot(F)

#ITALY
st.subheader("Disorders across Italy 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Italy.loc[:,~Italy.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Italy.index,Italy[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Italy.columns[8:10]):
    axs[i].plot(Italy.index,Italy[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
S= plt.figure(figsize=(8,6))
sb.heatmap(Italy.corr(), annot=True)
st.pyplot(S)

# LATVIA
st.subheader("Disorders across Latvia 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Latvia.loc[:,~Latvia.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Latvia.index,Latvia[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Latvia.columns[8:10]):
    axs[i].plot(Latvia.index,Latvia[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
F= plt.figure(figsize=(8,6))
sb.heatmap(Latvia.corr(), annot=True)
st.pyplot(F)

# LITHUANIA
st.subheader("Disorders across Lithuania 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Lithuania.loc[:,~Lithuania.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Lithuania.index,Lithuania[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Lithuania.columns[8:10]):
    axs[i].plot(Lithuania.index,Lithuania[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
h=plt.figure(figsize=(8,6))
sb.heatmap(Lithuania.corr(), annot=True)
st.pyplot(h)

# LUXEMBOURG
st.subheader("Disorders across Luxembourg 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Luxembourg.loc[:,~Luxembourg.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Luxembourg.index,Luxembourg[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Luxembourg.columns[9:11]):
    axs[i].plot(Luxembourg.index,Luxembourg[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
J=plt.figure(figsize=(8,6))
sb.heatmap(Malta.corr(), annot=True)
st.pyplot(J)

# HOLLAND
st.subheader("Disorders across Netherlands 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Netherlands.loc[:,~Netherlands.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Netherlands.index,Netherlands[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Netherlands.columns[8:10]):
    axs[i].plot(Netherlands.index,Netherlands[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
L= plt.figure(figsize=(8,6))
sb.heatmap(Netherlands.corr(), annot=True)
st.pyplot(L)

# POLAND

st.subheader("Disorders across Poland 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Poland.loc[:,~Poland.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Poland.index,Poland[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Poland.columns[8:10]):
    axs[i].plot(Poland.index,Poland[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
Z=plt.figure(figsize=(8,6))
sb.heatmap(Poland.corr(), annot=True)
st.pyplot(Z)

# PORTUGAL
st.subheader("Disorders across Portugal 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Portugal.loc[:,~Portugal.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Portugal.index,Portugal[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Portugal.columns[8:10]):
    axs[i].plot(Portugal.index,Portugal[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
v=plt.figure(figsize=(8,6))
sb.heatmap(Portugal.corr(), annot=True)
st.pyplot(v)

# CZECH REPUBLIC

st.subheader("Disorders across Czech Republic 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Czech_Republic.loc[:,~Czech_Republic.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Czech_Republic.index,Czech_Republic[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Czech_Republic.columns[8:10]):
    axs[i].plot(Czech_Republic.index,Czech_Republic[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
m=plt.figure(figsize=(8,6))
sb.heatmap(Czech_Republic.corr(), annot=True)
st.pyplot(m)

# ROMANIA

st.subheader("Disorders across Romania 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Romania.loc[:,~Romania.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Romania.index,Romania[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Romania.columns[8:10]):
    axs[i].plot(Romania.index,Romania[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
N=plt.figure(figsize=(8,6))
sb.heatmap(Romania.corr(), annot=True)
st.pyplot(N)

# SLOVENIA

st.subheader("Disorders across Slovenia 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()
colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Slovenia.loc[:,~Slovenia.columns.isin(['Country', 'Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Slovenia.index,Slovenia[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()
for i, col in enumerate(Slovenia.columns[8:10]):
    axs[i].plot(Slovenia.index,Slovenia[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
f=plt.figure(figsize=(8,6))
sb.heatmap(Slovenia.corr(), annot=True)
st.pyplot(f)

# SWEDEN
st.subheader("Disorders across Sweden 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(8, 15))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Sweden.loc[:,~Sweden.columns.isin(['Country','Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Sweden.index,Sweden[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# PREVALENCE DISTRIBUTION
st.markdown(":red[PREVALENCE IN GENDER]")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Sweden.columns[8:10]):
    axs[i].plot(Sweden.index,Sweden[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
st.pyplot(fig)

# correlation
st.markdown(":red[CORRELATION BETWEEN DISORDERS]")
z=plt.figure(figsize=(8,6))
sb.heatmap(Sweden.corr(), annot=True)
st.pyplot(z)

#EUROPE

Europe = pd.concat([Austria, Belgium, Bulgaria, Cyprus, Croatia, Denmark, Estonia, Finland, France, Germany, Greece, Slovakia, Spain, Hungary, Ireland, Italy, Latvia, Lithuania, Luxembourg, Malta, Netherlands, Poland, Portugal, Czech_Republic, Romania, Slovenia, Sweden], axis = 0)
Europe.info()
# groupby year: I want to create a dataset with the mean values of all the countries
Europe_df_ = Europe.groupby("Year")
Europe_df = Europe_df_.mean()
Europe_df.head()
Europe_df.info()

#PLOTS

st.subheader("Disorders across Europe 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(20, 20))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = Europe_df.loc[:,~Europe_df.columns.isin(['Country', 'Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(Europe_df.index,Europe_df[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
plt.show()

# PREVALENCE DISTRIBUTION
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(Europe_df.columns[7:9]):
    axs[i].plot(Europe_df.index,Europe_df[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
plt.show()

# correlation
plt.figure(figsize=(8,6))
sb.heatmap(Europe_df.corr(), annot=True)
plt.show()

#WORLD
World_df_ = mental_health_final.groupby("Year")
World_df = World_df_.mean()
World_df.info()

#PLOTS

st.subheader("Disorders across the World 1990-2017")
fig, axs = plt.subplots(4, 2, figsize=(20, 20))
axs = axs.ravel()

colors=['blue','red','green','purple','orange','brown','pink','gray']
columns_to_plot = World_df.loc[:,~World_df.columns.isin(['Country', 'Prevalence in males', 'Prevalence in females'])]

for i, col in enumerate(columns_to_plot):
    axs[i].plot(World_df.index,World_df[col],'o-',color=colors[i])
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
plt.show()

# PREVALENCE DISTRIBUTION
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs = axs.ravel()

for i, col in enumerate(World_df.columns[7:9]):
    axs[i].plot(World_df.index,World_df[col],'o-',color="red")
    axs[i].set_title(col)
    axs[i].set_xlabel('years')
    axs[i].set_ylabel('percentage')
plt.tight_layout()
plt.show()

# correlation
plt.figure(figsize=(8,6))
sb.heatmap(World_df.corr(), annot=True)
plt.show()

