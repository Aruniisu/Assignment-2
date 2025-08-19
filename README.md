# 🌸 Iris Classification API

A FastAPI-based machine learning API to classify iris flowers into
**Setosa**, **Versicolor**, or **Virginica** based on sepal and petal
dimensions.

------------------------------------------------------------------------

## 🚀 Features

-   Built with **FastAPI**
-   Uses a trained **RandomForestClassifier** (`model.pkl`)
-   REST API endpoints for prediction and model info
-   Returns prediction with **confidence score**

------------------------------------------------------------------------

## 📦 Installation

1.  Clone this repository:

    ``` bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  Create a virtual environment and install dependencies:

    ``` bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  Make sure you have `model.pkl` in the project folder.

------------------------------------------------------------------------

## ▶️ Running the API

Start the FastAPI server:

``` bash
uvicorn app:app --reload
```

API will run at:

    http://127.0.0.1:8000

------------------------------------------------------------------------

## 📡 API Endpoints

### 1. Health Check

``` http
GET /
```

**Response:**

``` json
{
  "status": "healthy",
  "message": "Iris Classification API is running"
}
```

------------------------------------------------------------------------

### 2. Predict Species

``` http
POST /predict
```

**Request Body:**

``` json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**

``` json
{
  "species": "setosa",
  "species_id": 0,
  "confidence": 0.98
}
```

------------------------------------------------------------------------

### 3. Model Info

``` http
GET /model-info
```

**Response:**

``` json
{
  "model_type": "RandomForestClassifier",
  "problem_type": "classification",
  "classes": ["setosa", "versicolor", "virginica"],
  "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  "accuracy": 1.0
}
```

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   Python
-   FastAPI
-   Scikit-learn
-   Joblib
-   Uvicorn

------------------------------------------------------------------------

## 📄 License

This project is licensed under the MIT License.
