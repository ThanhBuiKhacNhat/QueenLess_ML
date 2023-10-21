import os
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
import pickle
from .models import MODELS, get_model


def train_model(estimator, X_train, y_train, X_test, y_test):
    # Get the params grid for the current model
    dist = MODELS[estimator]['distribution']

    # Get model
    model = get_model(estimator)

    # Define evaluation
    cv = KFold(n_splits=10, shuffle=True, random_state=1)

    # Perform random search for the model
    n_iter = 30 if estimator != 'svm' else 10
    random_search = RandomizedSearchCV(
        model, param_distributions=dist, n_iter=n_iter, cv=cv,
        scoring='accuracy', verbose=3, n_jobs=4, random_state=1
    )

    # Fit the GridSearchCV object to the data.
    random_search.fit(X_train, y_train)

    # Get best score on training data and best model & best params
    train_accuracy = random_search.best_score_
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    # Evaluate on the test data
    test_accuracy = best_model.score(X_test, y_test)

    return train_accuracy, test_accuracy, best_model, best_params


def get_path(signature):
    # Get the absolute path to the root folder of the project
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create the path to the "export/signature" folder
    dataset_path = os.path.join(root_path, f"export/{signature}")

    # Check if this folder exists
    assert os.path.exists(dataset_path) and os.path.isdir(dataset_path)

    return dataset_path


def get_dataset(file_path):
    # Load the data from the csv file and split based on the 'is_test' column
    df = pd.read_csv(file_path)
    df_train = df[df['is_test'] == 0].drop(columns=['is_test'])
    df_test = df[df['is_test'] == 1].drop(columns=['is_test'])

    # Create the scaler object for StandardScaler
    scaler = StandardScaler()

    # Extract the features and labels from the df_train
    X_train = df_train.drop(columns=['label'])
    X_train = scaler.fit_transform(X_train)
    y_train = df_train['label']

    # Extract the features and labels from the df_test
    X_test = df_test.drop(columns=['label'])
    X_test = scaler.fit_transform(X_test)
    y_test = df_test['label']

    # Return the train and test set
    return X_train, y_train, X_test, y_test


def do_research(signature, dataset_file, model_list=MODELS.keys()):
    # Get dataset file
    export_path = get_path(signature)
    dataset_path = os.path.join(export_path, "data", dataset_file + '.csv')

    # Check if the file exists
    assert os.path.exists(dataset_path) and os.path.isfile(dataset_path)

    # Create folder for exporting the models
    model_path = os.path.join(export_path, 'models')

    # If this folder does not exist, create it
    if not os.path.exists(model_path) or not os.path.isdir(model_path):
        os.mkdir(model_path)

    # Create folder for exporting the results
    result_path = os.path.join(export_path, 'results')

    # If this folder does not exist, create it
    if not os.path.exists(result_path) or not os.path.isdir(result_path):
        os.mkdir(result_path)

    # Create the result dataframe
    result_columns = ['file', 'model', 'train_accuracy', 'test_accuracy']
    result_df = pd.DataFrame(columns=result_columns)

    # Load the dataset and split into train and test set
    X_train, y_train, X_test, y_test = get_dataset(dataset_path)

    # Create a log file for timing the training process
    log_file = os.path.join(result_path, f'logs_{dataset_file}.txt')
    with open(log_file, 'w') as f:
        f.write(f"Logs for {dataset_file} training:\n")

    # Iterate through all models
    for model in model_list:
        # Get name of the model
        model_name = MODELS[model]['name']

        # Train the model
        start_time = time.time()
        print(f"Training {model_name}")
        train_accuracy, test_accuracy, best_model, best_params = train_model(model, X_train, y_train, X_test, y_test)
        finish_time = time.time()
        elapsed_time = finish_time - start_time

        # Convert the elapsed time to hours, minutes, and seconds
        hours, remainder = divmod(int(elapsed_time), 3600)
        minutes, seconds = divmod(remainder, 60)

        # Open the log file in append mode
        with open(log_file, 'a') as f:
            f.write(f"\nTraining {model_name} done in {hours}:{minutes}:{seconds}\n"
                    f"Best params: {best_params}\n" + "-" * 50 + "\n")

        # Save the model to a file
        model_file = os.path.join(model_path, f"{model}.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(best_model, f)

        # Add new row to the result dataframe
        result_df.loc[len(result_df)] = [dataset_file, model_name, train_accuracy, test_accuracy]

        # Export the result dataframe to a csv file
        result_df.to_csv(os.path.join(result_path, f"results.csv"), index=False)

    print("Done training all models")
