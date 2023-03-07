import os
from urllib.request import urlretrieve
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

PKG_NAME = 'model_pkg.joblib'
MODEL_PKG_URL = 'https://raw.githubusercontent.com/JohnPPinto/Supply-Chain-Stock-Predictive-Analytics/main/model_pkg.joblib'

# Load dataset in pandas dataframe
def load_data(path: str):
    """
    This function takes in the path of the CSV file as a string
    and loads the file as a DataFrame.

    Parameters: path: str, relative path of the CSV file.

    Returns: df: pd.DataFrame, a pandas DataFrame of the CSV file. 
    """
    df = pd.read_csv(path)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    return df

# Download the joblib file containing all the objects
def download_pkg():
    """
    This function will download a joblib package containing objects
    for modeling and predicting the data.
    """
    urlretrieve(url=MODEL_PKG_URL, filename=PKG_NAME)
    print(f'\n[INFO] File Download to: {str(os.getcwd()) + "/" + PKG_NAME}\n')

# Training and Testing Algorithm
def train_test_algorithm(data: pd.DataFrame, train: bool=True):
    """
    This function takes in the dataframe to train or test the GradientBoosting
    model and displays the evaluated metrics durning the training stage, in the
    testing stage it returns the predicted values using the trained model.

    Parameters: data: pd.DataFrame, Data for the model training or testing.
                train: bool, True indicates model training and False indicates
                       model testing.
    
    Returns: pred: list, if testing is performed predicted values are returned.
    """
    # Downloading and loading the joblib package.
    urlretrieve(url=MODEL_PKG_URL, filename=PKG_NAME)
    model_pkg = joblib.load(PKG_NAME)
    
    # Scaling and one hot encoding the data
    df = data.copy()
    df[model_pkg['numeric_cols']] = model_pkg['scaler'].transform(df[model_pkg['numeric_cols']])
    df[model_pkg['encoded_cols']] = model_pkg['encoder'].transform(df[model_pkg['categorical_cols']])
    
    # Training the model
    if train:
        # Spliting the data in train and val set
        X_train, X_val, y_train, y_val = train_test_split(df[model_pkg['numeric_cols'] + model_pkg['encoded_cols']], 
                                                          df[model_pkg['target_col']], 
                                                          test_size=0.25, 
                                                          random_state=42)
        
        # Training the model and evaluating on the splited data.
        model = model_pkg['model']
        model.fit(X_train, y_train)
        train_rmse = mean_squared_error(y_train, model.predict(X_train), squared=False)
        val_rmse = mean_squared_error(y_val, model.predict(X_val), squared=False)
        train_mae = mean_absolute_error(y_train, model.predict(X_train))
        val_mae = mean_absolute_error(y_val, model.predict(X_val))
        print(f'\nMAE Result: Training: {train_mae:.6f}, Validation: {val_mae:.6f}')
        print(f'RMSE Result: Training: {train_rmse:.6f}, Validation: {val_rmse:.6f}\n')
    
    else:
        # Predicting the data using the model from the package
        X = df[model_pkg['numeric_cols'] + model_pkg['encoded_cols']]
        pred = model_pkg['model'].predict(X)
        print(f'[INFO] Predicted estimated stock percentage: {pred}')
        return pred.tolist()

# Execute the training and testing pipeline
def train_run(data_path: str):
    """
    This function executes the training pipeline by loading the prepared dataset
    from the CSV file and training the machine learning model.

    Parameters: data_path: str, relative path of the CSV file.
    """
    df = load_data(path=data_path)
    train_test_algorithm(data=df, train=True)

def test_run(data_path: str):
    """
    This function executes the testing pipeline by loading the prepared dataset
    from the CSV file and testing the machine learning model.

    Parameters: data_path: str, relative path of the CSV file.

    Returns: pred: list, Predicted Values.
    """
    df = load_data(path=data_path)
    pred = train_test_algorithm(data=df, train=False)
    return pred
