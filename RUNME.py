import sys
from src.audio_processing import generate_data
from src.training import do_research

# Change the signature for your experiment
# The data will be exported after processing and imported when training from the folder with this name
SIGNATURE = 'ENTER_YOUR_SIGNATURE_HERE'

do_generate_data = input('Enter "YES" to generate data: ')

# You can also enter signature and do_generate_data as command line arguments
if len(sys.argv) > 2:
    SIGNATURE = sys.argv[2]
    do_generate_data = sys.argv[3]

if do_generate_data == 'YES':
    generate_data(signature=SIGNATURE)

do_research(signature=SIGNATURE)
