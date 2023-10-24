import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_paths(signature, dataset_file):
    # Get the absolute path to the root folder of the project
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create the path to the "export/signature" folder
    export_path = os.path.join(root_path, "export")

    # Get the absolute path to the random forest models
    model_path = os.path.join(export_path, f"{signature}", "models", "random_forest.pkl")
    assert os.path.exists(model_path) and os.path.isfile(model_path)  # Check if the file exists

    # Get the absolute path to the original dataset
    dataset_path = os.path.join(export_path, f"{signature}", "data", f"{dataset_file}.csv")
    assert os.path.exists(dataset_path) and os.path.isfile(dataset_path)  # Check if the file exists

    # Get the path to the new export folder
    export_path = os.path.join(export_path, f"{signature}_fs")

    # Create folder for exporting the new signature folder
    if not os.path.exists(export_path) or not os.path.isdir(export_path):
        os.mkdir(export_path)

    # Get the path to the new dataset
    export_path = os.path.join(export_path, "data")

    # Create folder for exporting the new dataset
    if not os.path.exists(export_path) or not os.path.isdir(export_path):
        os.mkdir(export_path)

    # Return paths
    return model_path, dataset_path, export_path


def feature_selection(signature, dataset_file, n_select=10):
    # Get the random forest model's path
    model_path, dataset_path, export_path = get_paths(signature, dataset_file)

    # Load the model
    model = pickle.load(open(model_path, 'rb'))

    # Get the number of features in model
    n_features = model.n_features_in_
    assert n_features >= n_select

    # Create the feature names
    feature_names = [f"feature_{i + 1}" for i in range(n_features)]

    # Get feature importances from the model
    feature_importances = model.feature_importances_

    # Sort the features by importance
    sorted_idx = np.argsort(feature_importances)

    # Plot the feature importances of the forest
    plt.figure(figsize=(15, 9))
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Classifier - Feature Importance')

    # Save plot
    plt.savefig(os.path.join(export_path, f"{dataset_file}_feature_importance.png"), dpi=300)
    plt.show()

    # Get the top features
    top_feature_names = [feature_names[i] for i in sorted_idx[-n_select:]]

    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Extract the top features of the df
    df = df[top_feature_names + ['label', 'is_test']]

    # Export the new dataset
    df.to_csv(os.path.join(export_path, f"{dataset_file}.csv"), index=False)
