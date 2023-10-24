from src.audio_processing import generate_data
from src.training import do_research
from src.evaluate import evaluate
from src.feature_selection import feature_selection


# IMPORTANT ZONE: CONSTANTS FOR RUNTIME #

# Change the signature for your experiment
# The data will be exported after processing and imported when training from the folder with this name
SIGNATURE = 'ENTER_YOUR_SIGNATURE_HERE'

# True for generate new data, False for use existing data
GEN_DATA = False

# Segment length to crop audio files (in seconds)
SEGMENT_LENGTH = 2

# True for using overlapped segments, False for using non-overlapped segments
OVERLAP = False

# Set this to True for feature selection
FEATURE_SELECTION = True

# Number of features will be selected
N_SELECT = 10

# Training options: [logistic_regression, knn, decision_tree, random_forest, extra_trees, xgboost, svm]
MODEL_LIST = ['logistic_regression', 'knn', 'decision_tree', 'random_forest', 'extra_trees', 'xgboost', 'svm']

# Dataset options: [d_mean, d_std, d_merged, s_mean, s_std, s_merged]
DATASET = 's_merged'

# Set this to True for training
RESEARCH = False

# Set this to True for testing.
# Default: False (because the models already evaluated in the training process)
TEST = False

# END IMPORTANT ZONE #


if GEN_DATA:
    generate_data(signature=SIGNATURE, segment_length=SEGMENT_LENGTH, overlap=OVERLAP)

if FEATURE_SELECTION:
    feature_selection(signature=SIGNATURE, dataset_file=DATASET, n_select=N_SELECT)

if RESEARCH:
    if FEATURE_SELECTION:
        do_research(signature=SIGNATURE+'_fs', dataset_file=DATASET, model_list=MODEL_LIST)
    else:
        do_research(signature=SIGNATURE, dataset_file=DATASET, model_list=MODEL_LIST)

if TEST:
    evaluate(signature=SIGNATURE, dataset_file=DATASET)
