import pandas as pd  
from sklearn.model_selection import train_test_split, KFold  
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_absolute_error, r2_score  
from sklearn.pipeline import Pipeline  
from sklearn.compose import ColumnTransformer  
from sklearn.ensemble import StackingRegressor  


# Data Preprocessing  
def preprocess_data(food_data, glucose_data, activity_data):  
    """Preprocess and merge data from food, glucose, and activity datasets."""  
    
    merged_data = pd.merge(
        food_data, glucose_data, on=['user_id', 'timestamp'], how='inner'
    )  
    merged_data = pd.merge(
        merged_data, activity_data, on=['user_id', 'timestamp'], how='inner'
    )  

    # Feature Engineering  
    merged_data['glycemic_load'] = (
        merged_data['carbs'] * merged_data['glycemic_index'] / 100
    )  
    merged_data['meal_time'] = pd.to_datetime(
        merged_data['timestamp']
    ).dt.hour  

    return merged_data  


# Load datasets  
food_data = pd.read_csv('food_data.csv')  
glucose_data = pd.read_csv('glucose_data.csv')  
activity_data = pd.read_csv('activity_data.csv')  

# Preprocess the data  
data = preprocess_data(food_data, glucose_data, activity_data)  

# Verify columns  
if 'glucose_level' not in data.columns:  
    raise ValueError("Glucose level column is missing from the data.")  

# Split features and target  
X = data.drop(
    columns=['glucose_level', 'timestamp', 'user_id', 'glycemic_index']
)  
y = data['glucose_level']  

# Train-test split  
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  

# Preprocessing Pipeline for Scaling  
numeric_features = ['carbs', 'proteins', 'fats', 'glycemic_load', 'meal_time']  

numeric_transformer = Pipeline(
    steps=[('scaler', StandardScaler())]
)  

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features)]
)  

# Define Models  
model_rf = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100, random_state=42
        ))
    ]
)  

model_gb = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ]
)  

model_lr = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ]
)  

# Train and Evaluate RandomForest Model  
model_rf.fit(X_train, y_train)  
y_pred_rf = model_rf.predict(X_test)  

print(
    f"Random Forest MAE: {mean_absolute_error(y_test, y_pred_rf)}, "
    f"R²: {r2_score(y_test, y_pred_rf)}"
)  

# Train and Evaluate GradientBoosting Model  
model_gb.fit(X_train, y_train)  
y_pred_gb = model_gb.predict(X_test)  

print(
    f"Gradient Boosting MAE: {mean_absolute_error(y_test, y_pred_gb)}, "
    f"R²: {r2_score(y_test, y_pred_gb)}"
)  

# Train and Evaluate Linear Regression Model  
model_lr.fit(X_train, y_train)  
y_pred_lr = model_lr.predict(X_test)  

print(
    f"Linear Regression MAE: {mean_absolute_error(y_test, y_pred_lr)}, "
    f"R²: {r2_score(y_test, y_pred_lr)}"
)  

# Stacking Regressor (with KFold CV)  
stacking_model = StackingRegressor(
    estimators=[
        ('rf', model_rf), 
        ('gb', model_gb), 
        ('lr', model_lr)
    ],
    final_estimator=LinearRegression(),
    cv=KFold(n_splits=2)  # Adjusted to avoid splitting issues
)  

# Train and Evaluate Stacking Model  
stacking_model.fit(X_train, y_train)  
y_pred_stack = stacking_model.predict(X_test)  

print(
    f"Stacking Model MAE: {mean_absolute_error(y_test, y_pred_stack)}, "
    f"R²: {r2_score(y_test, y_pred_stack)}"
)  

# Output Sample Predictions  
print("Sample Predictions from Stacking Model:")  
print(y_pred_stack[:10])
