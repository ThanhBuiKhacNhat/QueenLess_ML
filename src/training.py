import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from .models import MODELS, get_model


def train_model(estimator, X_train, y_train, X_test, y_test):
    # Get the params grid for the current model
    dist = MODELS[estimator]['distribution']

    # Get model
    model = get_model(estimator)

    # Define evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # Perform grid search for the model
    random_search = RandomizedSearchCV(
        model, param_distributions=dist, n_iter=100, cv=cv,
        scoring='accuracy', verbose=0, n_jobs=-1, random_state=1
    )

    # Fit the GridSearchCV object to the data.
    random_search.fit(X_train, y_train)

    # Get best score on training data and best model & best params
    train_accuracy = random_search.best_score_
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    # Evaluate on the test data
    test_accuracy = best_model.score(X_test, y_test)

    return train_accuracy, test_accuracy, best_params


def get_paths(signature):
    # Get the absolute path to the root folder of the project
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create the path to the "export/signature" folder
    dataset_path = os.path.join(root_path, f"export/{signature}")

    # Check if this folder exists
    assert os.path.exists(dataset_path) and os.path.isdir(dataset_path)

    return dataset_path


def do_research(signature):
    dataset_path = get_paths(signature)

    # Create the result dataframe
    result_columns = ['file', 'model', 'train_accuracy', 'test_accuracy']

    result_df = pd.DataFrame(columns=result_columns)

    # Iterate from all csv files in the folder to load and train the model
    for file in os.listdir(dataset_path):
        if not file.endswith(".csv") or "result" in file:
            continue

        accepted = input(f'Enter "YES" to do research on {file}: ')

        if accepted != 'YES':
            continue

        print(f"Training on {file}")

        # Load the data from the csv file and split based on the 'is_test' column
        df = pd.read_csv(os.path.join(dataset_path, file))
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

        # Iterate through all models
        for model in MODELS.keys():
            # Get name of the model
            model_name = MODELS[model]['name']

            # Train the model
            print(f"Training {model_name}")
            train_accuracy, test_accuracy, best_params = train_model(model, X_train, y_train, X_test, y_test)

            # Print out result
            print(f"Train accuracy: {train_accuracy}")
            print(f"Test accuracy: {test_accuracy}")
            print(f"Best params: {best_params}")
            print(f"Done training {model_name}")

            # Add new row to the result dataframe
            result_df.loc[len(result_df)] = [file, model_name, train_accuracy, test_accuracy]

            # Export the result dataframe to a csv file
            result_df.to_csv(os.path.join(dataset_path, f"result.csv"), index=False)

    print("Done training")
