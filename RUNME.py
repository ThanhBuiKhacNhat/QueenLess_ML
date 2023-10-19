import sys
from src.audio_processing import generate_data
from src.training import do_research


# IMPORTANT ZONE: CONSTANTS FOR RUNTIME #

# Change the signature for your experiment
# The data will be exported after processing and imported when training from the folder with this name
SIGNATURE = 'non_overlapped_2s'

# True for generate new data, False for use existing data
GEN_DATA = False

# Segment length to crop audio files (in seconds)
SEGMENT_LENGTH = 2

# True for using overlapped segments, False for using non-overlapped segments
OVERLAP = True

# Dataset options: [d_mean, d_std, d_merged, s_mean, s_std, s_merged]
DATASET = 's_merged'

# END #


if GEN_DATA:
    generate_data(signature=SIGNATURE, segment_length=SEGMENT_LENGTH, overlap=OVERLAP)

do_research(signature=SIGNATURE, dataset_file=DATASET)
