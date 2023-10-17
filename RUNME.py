import sys
from src.audio_processing import generate_data
from src.training import do_research

# Change the signature for your experiment
# The data will be exported after processing and imported when training from the folder with this name
SIGNATURE = 'ENTER_YOUR_SIGNATURE_HERE'

# You can also specify your signature as command line's first argument
if len(sys.argv) > 1:
    SIGNATURE = sys.argv[1]

# You can enter your generate option by input or specify it as command line's second argument
if len(sys.argv) > 2:
    do_generate_data = sys.argv[2]
else:
    do_generate_data = input('Enter "YES" to generate data: ')

if do_generate_data == 'YES':
    generate_data(signature=SIGNATURE)

do_research(signature=SIGNATURE)
