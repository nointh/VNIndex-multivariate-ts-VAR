from flask import Flask, render_template, request
from statsmodels.tsa.vector_ar.var_model import VARResults
import numpy as np
import pickle
from datetime import datetime, timedelta
import pandas as pd
from statsmodels.tsa.stattools import acf


model = pickle.load(open('aic_model.pkl', "rb"))

def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    dataset = pd.read_csv('output.csv', index_col=0, parse_dates=True)
    dataset = dataset[['Oil price', 'S&P500', 'Gold price', 'VN Index']]
    min_date = dataset.index.min()
    max_date = dataset.index.max()   

    diff_df = dataset.diff().dropna()

    duration = int(request.form['duration'])
    start_date = datetime.strptime(
                     request.form['start_date'],
                     '%Y-%m-%d')
    lags = model.k_ar
    if (start_date+timedelta(days=duration) > max_date or start_date - timedelta(days=lags+1) < min_date):
        return render_template('after.html', data={ 'out_of_range': 1})
    
    start_index = dataset.index.searchsorted(start_date)
    test_start_index = dataset.index.searchsorted(start_date)
    test_end_index = dataset.index.searchsorted(start_date+timedelta(days=duration))
    df_test = dataset[test_start_index:test_end_index]

    
    model_start_date = start_date - timedelta(days=lags+1)
    model_end_date = start_date
    #  - timedelta(days=1)
    
    start_index = dataset.index.searchsorted(model_start_date)
    end_index = dataset.index.searchsorted(model_end_date)

    df_train = dataset[start_index:end_index]

    diff_df = dataset.diff().dropna()

    diff_start_index = diff_df.index.searchsorted(model_start_date)
    diff_end_index = diff_df.index.searchsorted(model_end_date)

    forecast_input = diff_df.values[diff_start_index:diff_end_index]

    fc = model.forecast(y=forecast_input, steps=duration)
    df_forecast = pd.DataFrame(fc, index=df_test.index, columns=dataset.columns + '_1d')
    df_results = invert_transformation(df_train, df_forecast, second_diff=False)
    df_results = df_results[['Oil price', 'S&P500', 'Gold price', 'VN Index']]

    frame = [df_train, df_results]
    full_result = pd.concat(frame)

    df_real = dataset[dataset.index.searchsorted(model_start_date): test_end_index]
    
    accuracy_vnindex = forecast_accuracy(df_results['VN Index'].values, df_test['VN Index'])
    accuracy_sp500 = forecast_accuracy(df_results['S&P500'].values, df_test['VN Index'])
    accuracy_gold = forecast_accuracy(df_results['Gold price'].values, df_test['VN Index'])
    accurac_oil = forecast_accuracy(df_results['Oil price'].values, df_test['VN Index'])
    
    accuracy_prod = {
        'VN Index': accuracy_vnindex,
        'S&P500': accuracy_sp500,
        'Gold price': accuracy_gold,
        'Oil price': accurac_oil
    }

    print('date', start_date)
    print('duration', duration)
    data = {
        'start_date': start_date,
        'duration': duration,
        'lags': model.k_ar,
        'accuracy': accuracy_prod,
        'true': df_real,
        'predict': df_results
    }
    # pred = model.predict(arr)
    return render_template('after.html', data=data)
if (__name__):
    app.run(debug=True)






