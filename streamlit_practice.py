import streamlit as st
import time

# packages for SARIMA
import pickle

# standard library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objects as go

# related third-party imports
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

from src.sarima_model import *

st.write("Hello ,let's learn how to build a streamlit app together")
st.title("this is the app title")
st.header("this is the markdown")
st.markdown("this is the header")
st.subheader("this is the subheader")
st.caption("this is the caption")
st.code("x=2021")
st.latex(r""" a+a r^1+a r^2+a r^3 """)

st.subheader("this a image")
st.image("streamlit_files/DSC_0422.JPG")
# st.audio("streamlit_files/Audio.mp3")
st.video("streamlit_files/Ah Bob video.mp4")

st.checkbox("yes")
st.button("Click")
st.radio("Pick your gender", ["Male", "Female"])
st.selectbox("Pick your gender", ["Male", "Female"])
st.multiselect("choose a planet", ["Jupiter", "Mars", "neptune"])
st.select_slider("Pick a mark", ["Bad", "Good", "Excellent"])
st.slider("Pick a number", 0, 50)

st.number_input("Pick a number", 0, 10)
st.text_input("Email address")
st.date_input("Travelling date")
st.time_input("School time")
st.text_area("Description")
st.file_uploader("Upload a photo")
st.color_picker("Choose your favorite color")

st.balloons()
st.progress(10)
with st.spinner("Wait for it..."):
    time.sleep(3)

st.success("You did it !")
st.error("Error")
st.warning("Warning")
st.info("It's easy to build a streamlit app")
st.exception(RuntimeError("RuntimeError exception"))

# side bar
st.sidebar.title("Testing side bitch")
# st.sidebar.button("Click") #cannot have 2 st.button in a single file
st.sidebar.radio("pick", ["gay", "lesbian"])

# container = st.container()
# container.write["This is written inside the container"]
# st.write["This is written outside the container"]


# plotting graphs
import matplotlib.pyplot as plt
import numpy as np

rand = np.random.normal(1, 2, size=20)
fig, ax = plt.subplots()
ax.hist(rand, bins=15)
st.pyplot(fig)

import pandas as pd

df = pd.DataFrame(np.random.randn(10, 2), columns=["x", "y"])
st.line_chart(df)

import graphviz as graphviz

st.graphviz_chart(
    """    
    digraph {        
        Big_shark -> Tuna        
        Tuna -> Mackerel        
        Mackerel -> Small_fishes        
        Small_fishes -> Shrimp    
        }"""
)


# load model and plot the results
def main():
    st.title("SARIMA Model Deployment")

    # load data
    data = load_data_full()
    data = pd.DataFrame(data)

    # Load the saved model from the pickle file
    with open("sarima_model_example.pkl", "rb") as file:
        loaded_model = pickle.load(file)
    # Make predictions using the loaded model
    forecast = loaded_model.forecast(steps=12)

    # Plot the forecasted values
    st.write("Forecasted values:")
    # fig, ax = plt.subplots()
    # ax.plot(data.index, data["num_orders"], label="Original Data")
    # ax.plot(forecast.index, forecast, label="Forecast")
    # ax.legend()
    # st.pyplot(fig)

    # # Display the predictions or any other desired output
    # st.write("Forecasted values:")
    # st.write(forecast)

    # Create Plotly figure for the forecasted values
    fig = go.Figure()

    # Add original data trace
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["num_orders"], mode="lines", name="Original Data"
        )
    )

    # Add forecasted values trace
    fig.add_trace(
        go.Scatter(x=forecast.index, y=forecast, mode="lines", name="Forecast")
    )

    fig.update_layout(
        title="Forecasted Values", xaxis_title="Date", yaxis_title="num_orders"
    )

    # Display the Plotly figure
    st.plotly_chart(fig)

    # Display the forecasted values
    st.write("Forecasted orders:")
    st.write(forecast)


if __name__ == "__main__":
    main()
