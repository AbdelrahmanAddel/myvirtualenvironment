# myvirtualenvironment - Disease Prediction API

This repository contains a Flask API for predicting diseases based on user-provided symptoms. It utilizes a pre-trained Support Vector Machine (SVM) classifier and a TF-IDF vectorizer to process and classify symptom text.

## Features

*   **Symptom-based Disease Prediction:** Predicts a disease based on textual input of symptoms.
*   **Flask API:** Provides a simple RESTful API endpoint for integration with other applications.
*   **Pre-trained Models:** Uses `best_svm_classifier.joblib` for classification and `tfidf_vectorizer.joblib` for text vectorization.
*   **Text Preprocessing:** Includes a basic text preprocessing function to clean symptom input.

## Technologies Used

*   **Python:** The core programming language.
*   **Flask:** A micro web framework for building the API.
*   **scikit-learn:** For machine learning models (SVM classifier and TF-IDF vectorizer).
*   **joblib:** For loading pre-trained models.

## Getting Started

To get this project up and running on your local machine, follow these steps:

### Prerequisites

Ensure you have Python installed. It is highly recommended to use a virtual environment.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/AbdelrahmanAddel/myvirtualenvironment.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd myvirtualenvironment
    ```
3.  Create and activate a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
4.  Install the required Python packages:
    ```bash
    pip install Flask scikit-learn joblib
    ```

### Running the API

To start the Flask API server, run the following command from the project directory:

```bash
python app.py
```

The API will run on `http://0.0.0.0:5000`.

## API Endpoint

### `/predict` (POST)

This endpoint accepts a JSON payload containing symptom text and returns the predicted disease.

*   **URL:** `/predict`
*   **Method:** `POST`
*   **Content-Type:** `application/json`

#### Request Body Example:

```json
{
    "text": "I have a cough and fever."
}
```

#### Response Body Example:

```json
{
    "prediction": "Common Cold"
}
```

#### Error Response Example:

```json
{
    "error": "No symptoms provided"
}
```

## Project Structure

*   `app.py`: The main Flask application file containing the API logic.
*   `best_svm_classifier.joblib`: The pre-trained SVM classifier model.
*   `tfidf_vectorizer.joblib`: The pre-trained TF-IDF vectorizer.
*   `README.md`: This README file.

## Contributing

Contributions are welcome! Please feel free to fork the repository, make improvements, and submit pull requests.

## License

This project is licensed under the MIT License.

## Contact

For any questions or suggestions, please contact [Abdelrahman Addel](https://github.com/AbdelrahmanAddel).


