import os
import shutil
import pandas as pd
from tqdm import tqdm
from pydub import AudioSegment
from .feature_extraction import calc_mfcc


def is_positive_integer(input_str):
    try:
        num = int(input_str)
        if num > 0:
            return True
        else:
            return False
    except ValueError:
        return False


def crop_audio(input_file, output_folder, segment_length):
    # List to store all features of all segments in this audio
    features = []

    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Calculate the total length of the audio in milliseconds
    total_length = len(audio)

    # Calculate the number of segments
    num_segments = total_length // segment_length

    # Get the name of the input file without the extension (.wav)
    input_file_name = os.path.splitext(os.path.basename(input_file))[0]

    # Crop and export each segment
    for i in tqdm(range(num_segments), desc=f"{input_file_name}", ascii=False, ncols=100):
        # Get the segment from the original audio
        start_time = i * segment_length
        end_time = (i + 1) * segment_length
        segment = audio[start_time:end_time]

        # Create the segment audio file
        output_file = os.path.join(output_folder, f"segment_{i + 1}.wav")

        # Export the temporary segment audio file
        segment.export(output_file, format="wav")

        # Calculate the MFCCs
        mfcc = calc_mfcc(output_file)

        # Append the mfcc to the features list
        features.append(mfcc)

        # Delete the segment audio file
        os.remove(output_file)

    return features


def crop_folder_audios(input_folder, output_folder, segment_length):
    # List to store all features, labels and is_tests of all audios in this folder
    features = []

    # Iterate through the items
    for file in os.listdir(input_folder):
        # Create the absolute path by joining the folder and file
        file_path = os.path.join(input_folder, file)

        # Check if it is a file
        if os.path.isfile(file_path):
            # Get the features from the audio file
            sub_features = crop_audio(file_path, output_folder, segment_length)

            # Concat the sub features and labels to the features and labels list
            features.extend(sub_features)

    return features


def get_paths(signature):
    # Get the absolute path to the root folder of the project
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Get the path to the "dataset" folder
    dataset_path = os.path.join(root_path, "dataset")

    # Create the path to the "export" folder
    export_path = os.path.join(root_path, "export")

    # If the "export" folder does not exist, create it
    if not os.path.exists(export_path) or not os.path.isdir(export_path):
        os.mkdir(export_path)

    # Create the path to the output folder for this signature
    output_path = os.path.join(export_path, signature)

    # Remove if the output folder already exists
    if os.path.exists(output_path) and os.path.isdir(output_path):
        shutil.rmtree(output_path)

    # Create the output folder
    os.makedirs(output_path)

    return dataset_path, output_path


def export_data(features, labels, is_tests, output_path):
    print("Exporting all features and labels to csv files.")

    # Get the list of all field's names in the AudioFeatures class
    key_list = []
    for key, value in vars(features[0]).items():
        key_list.append(key.replace('_AudioFeatures__', ''))

    # Create the dataframe for each type of features
    data = [[] for _ in range(len(key_list))]

    for i in range(len(features)):
        for key, value in vars(features[i]).items():
            # Get the index of the key (field name of AudioFeatures)
            index = list(vars(features[i]).keys()).index(key)

            # Append the value to the corresponding list
            data[index].append(value)

    # Export the dataframe for each type of features to csv files
    for i in range(len(key_list)):
        df = pd.DataFrame(data[i])
        # Rename the columns
        df.columns = [f'feature_{i}' for i in range(1, len(df.columns) + 1)]

        # Add the label (0-no_queen; 1-queen) and is_test (0-train; 1-test) columns
        df['label'] = labels
        df['is_test'] = is_tests

        # Export to the corresponding csv file
        df.to_csv(os.path.join(output_path, f"{key_list[i]}.csv"), index=False)

    print("Successfully export all features and labels to csv files.")


def generate_data(signature):
    # List to store all features, labels and is_tests of all folders in the dataset
    features = []
    labels = []
    is_tests = []

    # Get the absolute path to the "dataset" and "output" folder
    dataset_path, output_path = get_paths(signature)

    # Get the segment length for this folder from user
    user_input = input(f"Enter the segment length (in ms) for audios: ")
    while not is_positive_integer(user_input):
        if len(user_input) == 0:
            return

        print("The input is not a positive integer.")
        user_input = input(f"Enter the segment length (in ms) for audios: ")

    segment_length = int(user_input)

    # Iterate through all items
    for folder in os.listdir(dataset_path):
        # Create the absolute path by joining the folder and item
        folder_path = os.path.join(dataset_path, folder)

        # Check if it is not a folder
        if not os.path.isdir(folder_path):
            continue

        input_folder = os.path.basename(folder_path)

        # Start generate data
        print(f"Start generate data for {input_folder}.")

        # Get the features, labels and is_tests from the audios in this folder
        sub_features = crop_folder_audios(folder_path, output_path, segment_length)

        # Check if this folder is NoQueen or Queen; is_test or not
        label = 0 if "NoQueen" in folder else 1
        is_test = 0 if "test" not in folder else 1

        # Extend the features, labels and is_tests list
        features.extend(sub_features)
        labels.extend([label] * len(sub_features))
        is_tests.extend([is_test] * len(sub_features))

        print(f"Complete generate data for {input_folder} "
              f"with the segment length of {segment_length // 1000}s.")

    print("Complete generate data for all folders in the dataset.")

    # Export all features and labels to csv files
    export_data(features, labels, is_tests, output_path)
