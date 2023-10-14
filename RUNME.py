from src.audio_processing import generate_data
from src.training import do_research

# The data will be exported after processing and imported when training from the folder with this name
SIGNATURE = input('Enter signature: ').strip()

do_generate_data = input('Enter "YES" to generate data: ').strip()
if do_generate_data == 'YES':
    generate_data(signature=SIGNATURE)

do_research(signature=SIGNATURE)
