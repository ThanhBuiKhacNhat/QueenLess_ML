import os
import pandas as pd
import pickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def format_time(elapsed_time):
    # Convert the elapsed time to hours, minutes, and seconds
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours}:{minutes}:{seconds}"


def get_path(signature):
    # Get the absolute path to the root folder of the project
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create the path to the "export/signature" folder
    export_path = os.path.join(root_path, "export", f"{signature}")

    # Return the absolute path to datasets and models
    data_path = os.path.join(export_path, "data")
    model_path = os.path.join(export_path, 'models')
    return data_path, model_path


def get_test_data(file_path):
    df = pd.read_csv(file_path)
    df_test = df[df['is_test'] == 1].drop(columns=['is_test'])

    scaler = StandardScaler()
    X_test = df_test.drop(columns=['label'])
    X_test = scaler.fit_transform(X_test)
    y_test = df_test['label']

    return X_test, y_test


def evaluate(signature, dataset_file):
    # Get the absolute path to datasets and models
    data_path, model_path = get_path(signature)
    print(data_path)
    assert os.path.exists(data_path) and os.path.isdir(data_path)  # Check if the folder exists

    # Get the absolute path to the specific dataset
    csv_file = os.path.join(data_path, f'{dataset_file}.csv')
    print(csv_file)
    assert os.path.exists(csv_file) and os.path.isfile(csv_file)  # Check if the file exists

    # Get the list of all model files in the model folder
    model_files = [f for f in os.listdir(model_path) if
                   os.path.isfile(os.path.join(model_path, f)) and f.endswith('.pkl')]

    # Get the list of all model names
    model_names = [f.replace('.pkl', '') for f in model_files]

    # Create the dataframe for storing the results
    df = pd.DataFrame(columns=['model_name', 'dataset_name', 'accuracy', 'time'])

    # For each model, load the model and evaluate it on each dataset
    for model_name in model_names:
        print("Start evaluating model: {}".format(model_name))
        # Load the model
        model = pickle.load(open(os.path.join(model_path, model_name + '.pkl'), 'rb'))

        # Get the dataset
        X_test, y_test = get_test_data(os.path.join(data_path, csv_file))

        # Start timer
        start_time = time.time()

        # Predict the labels
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)

        # Finish timer
        end_time = time.time()

        # Append the result to the dataframe
        df.loc[len(df)] = [model_name, dataset_file, accuracy, format_time(end_time - start_time)]

        print("Finish evaluating model: {}".format(model_name))

    # Export the result to csv file
    df.to_csv(os.path.join(model_path, 'result.csv'), index=False)
