from copyreg import pickle
import streamlit as st
import sklearn
import pickle
import pandas as pd
import numpy as np
from PIL import Image

model = pickle.load(open('model.sav', 'rb'))

st.title('Diamond Price Predictor')
st.sidebar.header('Diamond Features')
image = Image.open('diamond.jpg')
st.image(image, 'Diamond pic')

#function
def user_report():
    carat = st.sidebar.slider('carat', 1, 7, 1)
    cut = st.sidebar.slider('cut', 0, 4, 1)
    clarity = st.sidebar.slider('clarity', 1, 7, 1)
    table = st.sidebar.slider('table', 1, 100, 1)
    depth = st.sidebar.slider('depth', 1, 100, 1)
    x = st.sidebar.slider('x', 50, 100, 1)
    y = st.sidebar.slider('y', 50, 100, 1)
    z = st.sidebar.slider('z', 50, 100, 1)

    user_report_data = {
        'carat': carat,
        'cut' : cut,
        'clarity' : clarity,
        'table' : table,
        'depth' : depth,
        'x' : x,
        'y' : y,
        'z' : z
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data


user_data = user_report()
st.header('Diamond Feature Data')
st.write(user_data)

price = model.predict(user_data)
st.subheader('Diamond Price')
st.subheader('$' + str(np.round(price[0], 4)))