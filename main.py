import threading
# import "packages" from flask
from flask import render_template,request  # import render_template from "public" flask libraries
from flask.cli import AppGroup
from flask import Flask
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


# import "packages" from "this" project
from __init__ import app, db, cors  # Definitions initialization

import json, jwt
from flask import Blueprint, request, jsonify, current_app, Response
from flask_restful import Api, Resource # used for REST API building
from datetime import datetime
from auth_middleware import token_required

from model.users import User
# import "packages" from "this" project
from __init__ import app, db, cors  # Definitions initialization

# setup APIs
from api.covid import covid_api # Blueprint import api definition
from api.joke import joke_api # Blueprint import api definition
from api.user import user_api # Blueprint import api definition
from api.player import player_api
# database migrations
from model.users import initUsers
from model.players import initPlayers
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# setup App pages
from projects.projects import app_projects # Blueprint directory import projects definition


# Initialize the SQLAlchemy object to work with the Flask app instance
db.init_app(app)
app = Flask(__name__)
CORS(app)

# register URIs
app.register_blueprint(joke_api) # register api routes
app.register_blueprint(covid_api) # register api routes
app.register_blueprint(user_api) # register api routes
app.register_blueprint(player_api)
app.register_blueprint(app_projects) # register app pages

@app.errorhandler(404)  # catch for URL not found
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('404.html'), 404

@app.route('/')  # connects default URL to index() function
def index():
    return render_template("index.html")

@app.route('/table/')  # connects /stub/ URL to stub() function
def table():
    return render_template("table.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Read the Iris dataset
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

    # Define the predict_variety function
    def predict_variety(sepal_length, sepal_width, petal_length, petal_width):
        # Make prediction
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        # Inverse transform the encoded prediction to get original variety
        predicted_variety = label_encoder.inverse_transform(prediction.astype(int))
        return predicted_variety[0]

    # Get data from request
    data = request.json
    sepal_length = data["sepal_length"]
    sepal_width = data["sepal_width"]
    petal_length = data["petal_length"]
    petal_width = data["petal_width"]
    
    # Call the predict_variety function
    predicted_variety = predict_variety(sepal_length, sepal_width, petal_length, petal_width)
    
    # Return the prediction as JSON response
    return jsonify({'predicted_variety': predicted_variety})


@app.before_request
def before_request():
    # Check if the request came from a specific origin
    allowed_origin = request.headers.get('Origin')
    if allowed_origin in ['http://localhost:4100/student3.0/', 'http://127.0.0.1:4100/student3.0/', 'https://nighthawkcoders.github.io', 'http://127.0.0.1:4100/student3.0//2024/03/21/iristesting.html']:
        cors._origins = allowed_origin


# Create an AppGroup for custom commands
custom_cli = AppGroup('custom', help='Custom commands')

# Define a command to generate data
@custom_cli.command('generate_data')
def generate_data():
    initUsers()
    initPlayers()

# Register the custom command group with the Flask application
app.cli.add_command(custom_cli)
        
# this runs the application on the development server
if __name__ == "__main__":
    # change name for testing
    app.run(debug=True, host="0.0.0.0", port="8086")
