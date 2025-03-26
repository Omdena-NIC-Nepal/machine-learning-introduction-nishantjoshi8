# Predicting House Prices Using Linear Regression

This project aims to predict house prices using linear regression techniques applied to the Boston Housing Dataset. The dataset contains various features related to housing and the target variable, which is the median home value.

## Project Structure

- **data/**: Contains the datasets used in the project.
  - **raw/**: Original dataset.
    - `boston_housing.csv`: The Boston Housing Dataset.
  - **processed/**: Cleaned and preprocessed dataset.
    - `processed_data.csv`: The dataset after preprocessing steps.

- **notebooks/**: Jupyter notebooks for different stages of the project.
  - `01_data_exploration.ipynb`: Data exploration and visualization.
  - `02_data_preprocessing.ipynb`: Data cleaning and preprocessing.
  - `03_model_building.ipynb`: Model training and hyperparameter tuning.
  - `04_model_evaluation.ipynb`: Model evaluation and performance metrics.
  - `05_feature_engineering.ipynb`: Feature engineering and testing.

- **src/**: Source code for the project.
  - `data_preprocessing.py`: Functions for data preprocessing tasks.
  - `model.py`: Functions for defining and training the linear regression model.
  - `evaluation.py`: Functions for evaluating model performance.
  - `feature_engineering.py`: Functions for creating and testing new features.

- **tests/**: Unit tests for the project.
  - `test_data_preprocessing.py`: Tests for data preprocessing functions.
  - `test_model.py`: Tests for model training and prediction functions.
  - `test_evaluation.py`: Tests for evaluation functions.

- **.gitignore**: Specifies files and directories to be ignored by version control.

- **requirements.txt**: Lists required Python packages and their versions.

- **LICENSE**: Licensing information for the project.

## Getting Started

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd Predicting-House-Prices-Using-Linear-Regression
   ```

2. **Install the required packages**:
   ```
   pip install -r requirements.txt
   ```

3. **Data Exploration**:
   Open `notebooks/01_data_exploration.ipynb` to explore the dataset and visualize relationships.

4. **Data Preprocessing**:
   Use `notebooks/02_data_preprocessing.ipynb` to clean and preprocess the data.

5. **Model Building**:
   Train the model using `notebooks/03_model_building.ipynb`.

6. **Model Evaluation**:
   Evaluate the model's performance in `notebooks/04_model_evaluation.ipynb`.

7. **Feature Engineering**:
   Experiment with new features in `notebooks/05_feature_engineering.ipynb`.

## License

This project is licensed under the MIT License. See the LICENSE file for details.