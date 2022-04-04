from orion import Orion
import yfinance as yf
import pandas as pd
import numpy as np


#Importig Apple Stock data from yfinance
appl_df = yf.download("AAPL", start="2019-01-01", end="2020-05-01")
print(appl_df)
#Getting Date and Adjusted Closing Price of Apple Stock
appl_df_1 = appl_df.reset_index()[['Date','Adj Close']]

#Making table with Date converted to timestamp in nanoseconds
tmdata = pd.Series(appl_df_1['Date'].values.astype(np.int64))

#Merging table with timestamp with original Table
appl_df_2 = appl_df_1.merge(tmdata.rename('Timestamp'), how = 'inner', left_index= True, right_index = True)

#Dropping Date
appl_df_2.drop('Date', axis=1, inplace= True)

#Converting table for model input
appl_df_3 = appl_df_2.reindex(columns = ['Timestamp', 'Adj Close'])
appl_final = appl_df_3.rename(columns={'Timestamp': 'timestamp', 'Adj Close': 'value'})
appl_final['value'] = appl_final['value'] /appl_final['value'].abs().max()
appl_final['timestamp'] = appl_final['timestamp'].div(1000000000)

#INPUT
print(appl_final)

#Tadgan
hyperparameters = {
    
    'orion.primitives.tadgan.TadGAN#1': {
        'epochs': 5
        }
}

orion = Orion(
    pipeline='tadgan',
    hyperparameters=hyperparameters
)

orion.fit(appl_final)

#Output
anomalies = orion.detect(appl_final)
print(anomalies)