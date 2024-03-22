import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('iris.csv')

# Encode the 'variety' column
label_encoder = LabelEncoder()
df['variety'] = label_encoder.fit_transform(df['variety'])

# Split the data into features and target
X = df.drop(columns=['variety'])
y = df['variety']

# Split data into train and test sets
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

def predict_variety(sepal_length, sepal_width, petal_length, petal_width):
    # Make prediction
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    # Inverse transform the encoded prediction to get original variety
    predicted_variety = label_encoder.inverse_transform(prediction.astype(int))
    return predicted_variety[0]
