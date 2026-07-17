import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("/mnt/user-data/uploads/job_salary_prediction_dataset.csv")

X = df.drop("salary", axis=1)
y = df["salary"]

X = pd.get_dummies(X, drop_first=True)
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("R2  :", r2_score(y_test, y_pred))

joblib.dump(model, "model/salary_prediction_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(feature_names, "model/feature_names.pkl")

# Also save the raw category options for building the frontend form
cat_cols = df.drop("salary", axis=1).select_dtypes(include="object").columns
options = {col: sorted(df[col].unique().tolist()) for col in cat_cols}
joblib.dump(options, "model/category_options.pkl")

num_cols = df.drop("salary", axis=1).select_dtypes(exclude="object").columns
ranges = {col: (int(df[col].min()), int(df[col].max())) for col in num_cols}
joblib.dump(ranges, "model/numeric_ranges.pkl")

print("Saved model, scaler, feature_names, category_options, numeric_ranges")
print(options)
print(ranges)
