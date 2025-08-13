import pandas as pd
import numpy as np
from math import radians,sin,cos,sqrt, atan2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# LOADING DATA
df= pd.read_csv('/Users/rajveersingh/Desktop/food_prediction/train.csv')


def haversine(lat1, lon1, lat2, lon2): # USING HAVERSINE FORMULA TO CALCULATE DISTANCE FROM REST TO DELIVERY
    R= 6371.0
    lat1, lon1, lat2, lon2= map(radians, [lat1, lon1, lat2, lon2])
    dlat= lat2- lat1
    dlon= lon2-lon1
    a= sin(dlat/2)**2 +cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c= 2* atan2(sqrt(a), sqrt(1-a))
    return R*c

df['distance_km']= df.apply(lambda row: haversine(row['Restaurant_latitude'],row['Restaurant_longitude'],row['Delivery_location_latitude'],row['Delivery_location_longitude']), axis=1) 
df = pd.get_dummies(df, columns=['Type_of_vehicle', 'Weatherconditions'], drop_first=True) # CONVERTING NON NUMERICAL VALUES TO BINARY FOR BETTER RESULTS
le = LabelEncoder()
df['Road_traffic_density'] = le.fit_transform(df['Road_traffic_density'])
df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%Y-%m-%d', errors='coerce')
df['Time_Orderd'] = pd.to_datetime(df['Time_Orderd'], format='%H:%M:%S', errors='coerce').dt.time
df['Time_Order_picked'] = pd.to_datetime(df['Time_Order_picked'], format='%H:%M:%S', errors='coerce').dt.time
# Remove "(min)" and convert to integer
df['Time_taken(min)'] = df['Time_taken(min)'].str.replace('(min)', '', regex=False).str.strip().astype(int)
df= df.replace('NaN', np.nan, regex=True)


def time_to_minutes(t):
    if pd.isnull(t):
        return None
    return t.hour * 60 + t.minute

df['Time_Orderd'] = df['Time_Orderd'].apply(time_to_minutes)
df['Time_Order_picked'] = df['Time_Order_picked'].apply(time_to_minutes)
df['order_to_pickup_time']= df['Time_Order_picked']- df['Time_Orderd']
df['Order_hour']= pd.to_datetime(df['Time_Orderd'], errors='coerce').dt.hour
df['Day_of_week']= df['Order_Date'].dt.dayofweek
columns_to_drop= ['multiple_deliveries','Festival','City','Delivery_person_ID','ID','Order_Date','Type_of_order'] # DROPPING COLUMNS NOT TO BE USED BY THE MODEL FOR PREDICTION
df= df.drop(columns=columns_to_drop)
# SPLITTING THE DATA
X= df.drop('Time_taken(min)', axis=1)
y= df['Time_taken(min)']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Step 10: Evaluate
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Step 11: Compare actual vs predicted
comparison_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
print(comparison_df)



