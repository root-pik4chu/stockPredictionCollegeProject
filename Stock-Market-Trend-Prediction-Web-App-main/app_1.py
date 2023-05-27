


















import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import streamlit.components.v1 as components



global df
st.title('Stock Analysis and Prediction ')

st.write("Stock prediction and analysis refers to the process of using historical stock data and machine learning algorithms to predict future stock prices and understand the underlying trends and patterns in the market.")
st.write("The stock market is constantly changing and affected by a wide range of factors such as company performance, economic indicators, and geopolitical events. By analyzing historical stock data, we can identify patterns and relationships that may help us make informed predictions about future stock prices.")
st.write("Machine learning algorithms like LSTM (Long Short-Term Memory) are particularly well-suited for this task because they can analyze large amounts of data and identify complex patterns that may be difficult for humans to recognize. For example, an LSTM algorithm could analyze historical stock prices, trading volumes, and other financial indicators to identify trends and patterns that may indicate a future price increase or decrease.")
st.write("Stock analysis can also help us understand the underlying factors that are driving changes in the market. By analyzing data on company financials, industry trends, and economic indicators, we can identify the key factors that are influencing stock prices and make more informed investment decisions.")
st.write("Overall, stock prediction and analysis is a valuable tool for investors and traders who want to make informed decisions about buying and selling stocks. By using historical data and machine learning algorithms to identify trends and patterns in the market, we can make more accurate predictions about future stock prices and gain a deeper understanding of the underlying factors that are driving market changes.")
# ======================================================================================================================================

if st.checkbox("Read More"):
    st.title("LSTM info")
    st.write("LSTM (Long Short-Term Memory) algorithm is a type of artificial intelligence that helps computers learn and make predictions based on patterns in data that change over time, such as stock prices, weather patterns, or speech.")
    st.write("Think of it like a person's memory - just like we remember events from the past to inform our decisions in the future, LSTM algorithms can remember important patterns in data from the past to make accurate predictions about what might happen in the future.")
    st.write("The LSTM algorithm is designed to understand which pieces of information from the past are most important to remember and which can be forgotten, just like how we tend to remember important moments in our lives but forget minor details. This helps the algorithm identify important trends and patterns in the data that can be used to make predictions.")
    st.write("Overall, LSTM is a powerful tool for analyzing and predicting time-series data, and can be used in a wide range of applications such as predicting stock prices, identifying anomalies in sensor data, and even generating realistic language or music.")
    # image .... 
    # image_file = "th.jpg"
    # st.image(image_file, caption="Optional image caption")

    image_file = "th.jpg"

    # Create a container to center the image
    container = st.container()
    with container:
        st.image(image_file, caption="LSTM structure", use_column_width=True, width=300, output_format='JPEG', channels='RGB')
        

    # Apply CSS to center the container
    container.markdown(
        f"""
        <style>
        .st-bt {{
            text-align: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.write("The image shows a diagram of a basic Long Short-Term Memory (LSTM) model, which consists of several parameters that work together to regulate the flow of information in the network. Here's a brief explanation of each parameter:")
    st.write("1.Input layer: This is where the input data is received by the LSTM network. The size of the input layer depends on the size of the input data.")
    st.write("2.LSTM layer: This is the main layer of the LSTM model that contains three gates: input gate, forget gate, and output gate. These gates control the flow of information in and out of the memory cell.")
    st.write("3.Memory cell: This is the core of the LSTM network, where information is stored over time.")
    st.write("4.Input gate: This gate controls the flow of new input data into the memory cell. It is activated by the sigmoid function, which regulates the flow of information between 0 and 1.")
    st.write("5.Forget gate: This gate controls which information should be removed from the memory cell. It is also activated by the sigmoid function.")
    st.write("6.Output gate: This gate controls the flow of output data from the memory cell. It is activated by the hyperbolic tangent (tanh) function, which regulates the flow of information between -1 and 1.")
    st.write("7.Hidden state: This is the output of the LSTM layer, which is a combination of the input data and the memory cell state.")
    st.write("8.Output layer: This is the final layer of the LSTM model that generates the prediction based on the input data and the hidden state.")
    st.write("These parameters work together to create a powerful machine learning model that can learn and predict patterns in sequential data.")
    st.write("")

    




# =======================================================================================================================================


# side barr ....

st.sidebar.subheader("visualization settings ")
uploaded_file = st.sidebar.file_uploader(label="upload csv file here",
                         type=['csv'])

# -------------------------------------------button----------


# Define the HTML code for the new page

# Define the button that will navigate to the new page
import subprocess

# if st.button('Open Jupyter Notebook'):
#     # Replace "/path/to/notebook.ipynb" with the actual path to your notebook file
#       subprocess.Popen(['jupyter', 'notebook', 'NewModel.ipynb'])
# import subprocess

def open_notebook(path):
    subprocess.Popen(['jupyter', 'notebook', 'NewModel.Ipynb'])
# -----------------button----------------------

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        

        st.write(df.head())
        #VISUALISATION 
        st.subheader('Closing Price vs Time chart')
        fig = plt.figure(figsize = (12,6))
        plt.plot(df.Close)
        st.pyplot(fig)





        st.subheader('Closing Price vs Time chart with 100MA')
        ma100 = df.Close.rolling(100).mean()
        fig = plt.figure(figsize = (12,6))
        plt.plot(ma100)
        plt.plot(df.Close)
        st.pyplot(fig)





        st.subheader('Closing Price vs Time chart with 100MA & 200MA')
        ma100 = df.Close.rolling(100).mean()
        ma200 = df.Close.rolling(200).mean()
        fig = plt.figure(figsize = (12,6))
        plt.plot(ma100,'r')
        plt.plot(ma200,'g')
        plt.plot(df.Close,'b')
        st.pyplot(fig)



        #Splitting Data into Training and Testing

        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.75)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.25): int(len(df))])

        print(data_training.shape)
        print(data_testing.shape)

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))


        data_training_array = scaler.fit_transform(data_training)



        #splitting data into x_train and y_train

        x_train = []
        y_train = []

        for i in range(100, data_training_array.shape[0]):
            x_train.append(data_training_array[i-100: i])
            y_train.append(data_training_array[i, 0])
            
        x_train, y_train = np.array(x_train), np.array(y_train)



        #load my model

        model = load_model('keras_model.h5')


        #testing part
        past_100_days = data_training.tail(100)
        final_df = past_100_days.append(data_testing, ignore_index = True)
        input_data = scaler.fit_transform(final_df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100: i])
            y_test.append(input_data[i, 0])




        #Making Predicitons
        x_test, y_test = np.array(x_test), np.array(y_test)
        y_predicted = model.predict(x_test)
        scaler = scaler.scale_

        scale_factor = 1/scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor


        #Final graph

        st.subheader('Predictions vs Original')
        fig2 = plt.figure(figsize=(12,6))
        plt.plot(y_test, 'b', label= 'Original Price')
        plt.plot(y_predicted, 'r', label = 'Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)



    except Exception as e:
        print(e)

try:
    st.write(df.describe())
except Exception as e:
    print(e)
    st.subheader('please upload the file for Results')
    # st.write("please upload the file ....")







# start = '2010-01-01'
# end = '2021-12-31'



# st.title('Stock Trend Prediction')

# user_input = st.text_input('Enter Stock Ticker', 'AAPL')
# df = data.DataReader(user_input, 'yahoo', start, end)


# #Describing Data
# st.subheader('Data from 2010 - 2021')

