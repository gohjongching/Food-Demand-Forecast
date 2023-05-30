# Model 2: Long-Short-Term-Memory (LSTM) Model

## Brief intro to LSTM
- RNNs are a type of neural network architecture that is designed to handle sequential data by maintaining a memory of previous inputs. In RNNs, the output of a hidden layer is fed back to the input of the same layer in the next time step. However, the main disadvantage of RNNs is the vanishing gradient problem, where the gradients of the earlier time steps become smaller and smaller, which results in difficulty in training the network and retaining long-term memory.

- LSTMs are a type of RNN that overcomes the vanishing gradient problem by introducing a memory cell and three gating mechanisms, namely the input gate, forget gate, and output gate. The memory cell helps the network to remember important information over longer time periods, while the gating mechanisms control the flow of information into and out of the cell. The forget gate decides which information should be discarded from the cell and the input gate decides which new information should be added to the cell. The output gate decides the output of the cell at the current time step.

<img src="https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/614fc586739fea6fc08532cf_lstm.png"  width="450" height="600">

- In this project, we are using LSTM to predict the demand for the next 10 weeks (week: 146 to 155). Note that to compare against the baseline model and time constraint, we will only be using univariate data (target only). Future work will include RNN with multivariate data. 

## Training an LSTM model on our food demand data involves the following steps:

1. Data Preprocessing: The data is first preprocessed to remove any outliers or missing values. Using our 'datapipeline.py', we will preprocess the data. 

2. Windowing: The data is then windowed, which involves creating overlapping sequences of data points to be used as inputs to the LSTM. We use the 'preprocess_window.py' to generate a normalized train and test torch.tensor dataset. The lookback and lookforward for this project is both 10 respectively (train on past 10 weeks data and predict advance 10 weeks data). 

3. Model Definition: The LSTM RNN model is defined, specifying the number of layers, the number of neurons in each layer, and the activation function to be used. 
    - The number of LSTM cell used here is 50 and the number of layers is 1.
        - Further hyperparameter tuning can be done by adjusting the hidden_size and num_layers. 
    - nn.Linear() is used here as Linear activation function simply returns the weighted sum of inputs plus a bias term, and is useful for regression problems where the output is a continuous value.
    - <pre><code>
    class LSTM(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    </code></pre>

4. Model Training: The model is then trained on the windowed data using an appropriate optimization algorithm, such as Adam.
    - Further hyperparameter tuning can be done by using different optimizer like RMSProp, Adagrad, SGD. 
5. Model Evaluation: The trained model is then evaluated on both test and train dataset to measure its performance and to determine if any further modifications to the model are necessary. In this project, we are using RMSE to evaluate our model.
    - Results:
        - <img src="https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/614fc586739fea6fc08532cf_lstm.png"  width="450" height="600">
        - <img src="https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/614fc586739fea6fc08532cf_lstm.png"  width="450" height="600">
        - Test RMSE:  96430.12. Train RMSE:  9501.999





