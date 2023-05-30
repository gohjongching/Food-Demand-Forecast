# standard library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# related third-party imports
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# local application/library specific imports
from .datapipeline import run_datapipeline


class SarimaModel:
    """
    A class to fit and evaluate a SARIMA model for time series forecasting.
    """

    def __init__(self, ts, test, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)):
        """
        Initialize a SarimaModel instance with the given time series data and hyperparameters.

        :param ts: the training time series data as a numpy array or pandas Series.
        :param test: the test set data as a numpy array or pandas Series.
        :param order: the non-seasonal order of the SARIMA model as a tuple of (p, d, q).
        :param seasonal_order: the seasonal order of the SARIMA model as a tuple of (P, D, Q, s).
        """
        self.ts = ts
        self.test = test
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.predictions = None
        self.eval_metric = None

    def fit(self, model_name=None):
        """
        Fit a SARIMA model to the time series data.
        """
        self.model = SARIMAX(self.ts, order=self.order,
                             seasonal_order=self.seasonal_order).fit()

        # Save the fitted SARIMA model to a file.
        if model_name is not None:
            model_name = Path(model_name)
            model_name.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(model_name)

    def save_model(self, model_name="models/sarima_model.pkl"):
        """
        Save the fitted SARIMA model to a file.

        :param model_name: the name of the file to save the model to.
        """
        model_name = Path(model_name)
        model_name.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_name)

    def predict(self, n_periods=10):
        """
        Generate predictions using the fitted SARIMA model.

        :param n_periods: the number of periods to forecast.
        :return: an array of predicted values based on the in-sample or out-of-sample predictions of the fitted SARIMA model.
        """
        return self.model.predict(start=len(self.ts), end=len(self.ts)+n_periods-1)

    def evaluate(self, y_true, y_pred):
        """
        Calculate the evaluation metric for the SARIMA model.

        :param y_true: the true values of the test set.
        :param y_pred: the predicted values of the test set.
        :return: the evaluation metric as a float.
        """
        self.eval_metric = np.sqrt(mean_squared_error(y_true, y_pred))
        return self.eval_metric

    def forecast(self, n_periods=10):
        """
        Generate forecasts using the fitted SARIMA model.

        :param n_periods: the number of periods to forecast.
        :return: an array of predicted values based on the in-sample or out-of-sample predictions of the fitted SARIMA model.
        """
        self.predictions = self.predict(n_periods=n_periods)
        return self.predictions

    def plot_forecast(self):
        """
        Plot the test and forecast values of the SARIMA model.
        """
        fig, ax = plt.subplots(figsize=(20, 3))
        ax.plot(self.test.index, self.test.values.tolist(), label='Actual')
        ax.plot(self.test.index, self.predictions, label='Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title('Actual vs. Forecast')
        ax.legend()
        plt.show()

    def rolling_forecast(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52), seasonal=False):
        """
        Generate rolling forecasts using the SARIMA model.

        :param order: the non-seasonal order of the SARIMA model as a tuple of (p, d, q).
        :param seasonal_order: the seasonal order of the SARIMA model as a tuple of (P, D, Q, s).
        :param seasonal: a boolean indicating whether to fit a seasonal SARIMA model.
        :return: a list of predicted values based on the rolling forecasts of the fitted SARIMA model.
        """
        lookback = [x for x in self.ts]
        predictions = list()

        # walk-forward validation
        for t in range(len(self.test)):
            self.model = SARIMAX(lookback, order=self.order,
                                 seasonal_order=self.seasonal_order)
            model_fit = self.model.fit()
            output = model_fit.forecast()
            lookahead = output[0]

            # manually keep track all observations - training data + new observations are appended each iteration
            predictions.append(lookahead)
            obs = self.test[t]
            lookback.append(obs)
            # print('predicted=%f, expected=%f' % (lookahead, obs))

        self.predictions = predictions

        return self.predictions


def load_data_full():
    """
    Load the full data and resample it by week.

    :return: a pandas Series representing the resampled time series data.
    """
    # Load the data using the pipeline function
    df_train, df_test = run_datapipeline()

    # Combine the training and test dataframes
    df = pd.concat([df_train, df_test])

    # Resample the DataFrame by week and sum the 'num_orders' column
    df_full = df['num_orders'].resample('W').sum()

    return df_full


def split_data(df):
    """
    Split the time series data into training and test sets.

    :param df: the time series data as a pandas Series.
    :return: a tuple of (training set, test set).
    """
    # Split the last 10 records as test series
    test = df[-10:]
    ts = df[:-10]

    return ts, test


def sarima_forecast(order=(1, 1, 1), seasonal_order=(1, 1, 1, 52), rolling=True, model_name=None):
    """
    Generate SARIMA forecasts for the time series data.

    :param order: the non-seasonal order of the SARIMA model as a tuple of (p, d, q).
    :param seasonal_order: the seasonal order of the SARIMA model as a tuple of (P, D, Q, s).
    :param model_name: the name of the file to save the fitted model to.
    :param rolling: a boolean indicating whether to generate rolling forecasts.
    :return: a SarimaModel instance representing the fitted SARIMA model.
    """
    # Load the data using the pipeline function
    df = load_data_full()
    ts, test = split_data(df)

    # Create and fit model
    model = SarimaModel(ts, test, order=order, seasonal_order=seasonal_order)
    model.fit(model_name)

    # Forecast
    if rolling:
        forecast = model.rolling_forecast(order, seasonal_order, seasonal=True)
    else:
        n_periods = test.shape[0]
        forecast = model.forecast(n_periods)

    # Evaluate
    eval_metric = model.evaluate(test, forecast)
    print('Evaluate metric:', eval_metric)

    # Visualize
    model.plot_forecast()

    return model
