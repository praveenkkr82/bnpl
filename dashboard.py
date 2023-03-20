import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.set_page_config( page_title="Dashboard", page_icon="🧊", layout="wide", initial_sidebar_state="expanded")
st.title("​Book Now Pay Later")


def extract():
    # extract data from database

    df = pd.read_csv('BNPL_data.csv')
    return df

def model_fit(df):

    df_dep = df.drop("Cost Matrix(Risk)", axis=1)
    df_ind = df["Cost Matrix(Risk)"]

    obj = [i for i in df_dep.columns if df_dep[i].dtype == "O"]
    df_dep_obj = pd.get_dummies(df[obj])
    num = [i for i in df_dep.columns if df_dep[i].dtype != "O"]

    df_dep = pd.concat([df_dep_obj, df[num]], axis=1)
    df_ind = np.where(df_ind=="Good Risk", 1, 0)
    df_de = pd.DataFrame()
    for i in sorted(df_dep.columns):
        df_de[i] = df_dep[i]

    #model = RandomForestClassifier()
    model = pickle.load(open('randomforest_model.pkl', 'rb'))
    model.fit(df_de, df_ind)

    return model, df_de.iloc[1:2,:]


if __name__=="__main__":
    df = extract()
    actual_model, df_dep = model_fit(df)

    df_col = [i for i in df.columns if df[i].dtypes == "O"]

    df_predict1 = pd.DataFrame()
    for i in range(len(df_col)-1):
        exec(f"s_{i+1} = st.selectbox(df_col[{i}], (j for j in df[df_col].iloc[:,{i}].unique()))")
        print(exec(f"s_{i+1}"))
        df_select = pd.DataFrame(dict ([(df_col[i]+'_'+m,1 if eval(f"s_{i+1}") == m else 0 ) for m in df[df_col].iloc[:,i].unique()]), index=[0])
        df_predict1 = pd.concat([df_predict1, df_select], axis=1)
    

    df_int = [i for i in df.columns if df[i].dtypes != "O"]
    d2 = dict()
    for k in range(len(df_int)):

        exec(f"s_{k+1+len(df_col)} = st.number_input(df_int[{k}],step=1, min_value=df[df_int[{k}]].min(), max_value=df[df_int[{k}]].max(), value = df[df_int[{k}]][500])")
        exec(f"d2[df_int[{k}]] = s_{k+1+len(df_col)}")

    df_predict2 = pd.DataFrame(d2, index=[0])
    df_predict = pd.concat([df_predict1, df_predict2], axis=1)
    
    print(sorted(df_predict))
    df_pre = pd.DataFrame()
    for i in sorted(df_dep.columns):
        df_pre[i] = df_predict[i]

    if st.button("Predict"):
        st.write("Prediction: Good Risk" if 1==actual_model.predict(df_pre)[0] else "Prediction: Bad Risk")
