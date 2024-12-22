import warnings; 
warnings.simplefilter('ignore')

!pip install pystan~=2.14
!pip install prophet

#Install necessary libraries
import warnings
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')

# Load and view the dataset

df = pd.read_csv('dataset.csv')
print(df.head())
print(df.describe())

# Check the dataset

df.dtypes
df.head()

# Convert 'Time Date' to a proper datetime format 
df['Year'] = df['Time Date'].apply(lambda x: str(x)[-4:])  
df['Month'] = df['Time Date'].apply(lambda x: str(x)[-6:-4])  
df['Day'] = df['Time Date'].apply(lambda x: str(x)[:-6])

# Combine all to a proper YYYY-MM-DD Format and drop unnecessary columns
df['ds'] = pd.to_datetime(df['Year'] + '-' + df['Month'] + '-' + df['Day'])
df.drop(['Time Date', 'Product', 'Store', 'Year', 'Month', 'Day'], axis=1, inplace=True)
df.columns = ['y', 'ds']

# Initialise and fit the prophet model

m = Prophet(interval_width=0.95, daily_seasonality=True)
model = m.fit(df)

# Make future predictions

future = m.make_future_dataframe(periods=100,freq='D')
forecast = m.predict(future)
forecast.head()

#Plotting the forecast
plot1 = m.plot(forecast)
plt2 = m.plot_components(forecast)

# Actual vs Predicted Value plotting

plt.plot(df["ds"],df["y"], label="Actual Values",color="red")
plt.plot(forecast["ds"],forecast["yhat"],label="Predicted Values",color="black")
plt.show()


,
