<<<<<<< HEAD
# food_delivery_time_prediction
=======
# Food Delivery Time Prediction

This project predicts food delivery time using a machine learning algorithm. The model estimates delivery time based on features like distance, vehicle type, weather conditions, and order timings.

---

## Dataset

The dataset includes order details,driver details such as restaurant and delivery locations, order and pickup times, vehicle types, weather, and other relevant factors.  
Since the dataset did not provide the distance between restaurant and delivery locations, the Haversine formula was used to calculate this as a new feature.

---

## Data Preprocessing

- Converted categorical variables (`Type_of_vehicle`, `Weatherconditions`) into binary features using one-hot encoding.  
- Encoded `Road_traffic_density` using label encoding.  
- Parsed `Order_Date`, `Time_Orderd`, and `Time_Order_picked` into datetime objects, then transformed times into numerical features (minutes from midnight).  
- Calculated delivery distance with the Haversine formula.  
- Dropped less relevant columns like `Vehicle_condition`, `multiple_deliveries`, `Festival`, `City`, `Delivery_person_ID`, `ID`, `Order_Date`, and `Type_of_order`.

---

## Model

A Random Forest Regressor was trained to predict delivery time (`Time_taken(min)`) using the processed features.

---

## Performance Metrics

- Mean Absolute Error (MAE): 3.3526176115802175
- Root Mean Squared Error (RMSE): 4.219768002218018
- R² Score: 0.7969112636640425

---

## How to Run

1. Clone the repository:  
   ```bash git clone https://github.com/rajwhezee/food_delivery_time_prediction.git
2. Navigate to the project folder:
    cd food_delivery_time_prediction
3. Install the required dependencies:
    pip install -r requirements.txt
4. Run the model training script:
    python train_food.py

## What to Expect

When you run `train_food.py`, the script will:

- Load and preprocess the food delivery dataset  
- Train a Random Forest regression model to predict delivery time  
- Print evaluation metrics to the console:
  - Mean Absolute Error (MAE)  
  - Root Mean Squared Error (RMSE)  
  - R² Score  
- Display a comparison table of actual vs predicted delivery times for the test dataset

No files are saved automatically, but you can modify the script to save models or plots if needed.

>>>>>>> 554bfff (adding readme, training code, dataset and requirements)
