from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def detect_date_column(data):
    # Try to find a date-like column
    for col in data.columns:
        try:
            pd.to_datetime(data[col])
            return col
        except:
            continue
    return None

def detect_numeric_columns(data):
    # Find all numeric columns
    return data.select_dtypes(include=['number']).columns.tolist()

def detect_categorical_columns(data):
    # Find all categorical columns
    return data.select_dtypes(include=['object', 'category']).columns.tolist()

def preprocess_data(data, preprocessing_steps):
    # Handle missing values
    if 'fill_missing_mean' in preprocessing_steps:
        data.fillna(data.mean(), inplace=True)
    elif 'fill_missing_median' in preprocessing_steps:
        data.fillna(data.median(), inplace=True)
    elif 'drop_missing' in preprocessing_steps:
        data.dropna(inplace=True)
    
    # Encode categorical variables
    if 'one_hot_encode' in preprocessing_steps:
        categorical_columns = detect_categorical_columns(data)
        if categorical_columns:
            data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    # Normalize/scale numeric data
    if 'normalize' in preprocessing_steps:
        numeric_columns = detect_numeric_columns(data)
        if numeric_columns:
            scaler = StandardScaler()
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    return data

def perform_analysis(data, option):
    try:
        # Detect date and numeric columns
        date_column = detect_date_column(data)
        numeric_columns = detect_numeric_columns(data)

        if option == "summary":
            return data.describe().to_html()
        
        elif option == "correlation":
            if len(numeric_columns) < 2:
                return "Error: At least two numeric columns are required for correlation analysis."
            return data[numeric_columns].corr().to_html()
        
        elif option == "trend":
            if not date_column or not numeric_columns:
                return "Error: Date and numeric columns are required for trend analysis."
            # Use the first numeric column for trend analysis
            trend_column = numeric_columns[0]
            plt.figure(figsize=(10, 6))
            plt.plot(data[date_column], data[trend_column])
            plt.title(f"{trend_column} Trend Over Time")
            plt.xlabel(date_column)
            plt.ylabel(trend_column)
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            return f"<img src='data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}'>"
        
        elif option == "cluster":
            if len(numeric_columns) < 2:
                return "Error: At least two numeric columns are required for clustering."
            cluster_columns = numeric_columns[:2]
            X = data[cluster_columns]
            kmeans = KMeans(n_clusters=3)
            data['Cluster'] = kmeans.fit_predict(X)
            
            # Plot clusters
            plt.figure(figsize=(10, 6))
            plt.scatter(data[cluster_columns[0]], data[cluster_columns[1]], c=data['Cluster'], cmap='viridis')
            plt.title(f"Clustering: {cluster_columns[0]} vs {cluster_columns[1]}")
            plt.xlabel(cluster_columns[0])
            plt.ylabel(cluster_columns[1])
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            return f"<img src='data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}'>"
        
        elif option == "predict":
            if not numeric_columns:
                return "Error: Numeric columns are required for prediction."
            # Use the first numeric column for prediction
            target_column = numeric_columns[0]
            if date_column:
                data['Days'] = (data[date_column] - data[date_column].min()).dt.days
                X = data[['Days']]
            else:
                X = data[numeric_columns[1:]]  # Use other numeric columns as features
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Plot predictions
            plt.figure(figsize=(10, 6))
            plt.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Actual')
            plt.scatter(X_test.iloc[:, 0], y_pred, color='red', label='Predicted')
            plt.title(f"{target_column} Prediction")
            plt.xlabel(X.columns[0])
            plt.ylabel(target_column)
            plt.legend()
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            return f"<img src='data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}'>"
        
        else:
            return "Invalid option selected."
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file is uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save the file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            try:
                # Load the data
                data = pd.read_csv(filepath)
                
                # Get the selected task
                task = request.form.get('task')
                
                if task == "preprocessing":
                    # Get preprocessing steps
                    preprocessing_steps = request.form.getlist('preprocessing')
                    if preprocessing_steps:
                        data = preprocess_data(data, preprocessing_steps)
                    result = "Preprocessing completed successfully."
                else:
                    # Get the selected analysis option
                    option = request.form.get('option')
                    result = perform_analysis(data, option)
                
                return render_template("index.html", result=result, filename=file.filename, task=task)
            
            except Exception as e:
                return render_template("index.html", result=f"Error: {str(e)}", filename=None, task=None)
    
    return render_template("index.html", result=None, filename=None, task=None)

if __name__ == "__main__":
    app.run(debug=True)