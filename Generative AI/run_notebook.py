"""Execute generative_model.ipynb in place."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

here = Path(__file__).parent
nb_path = here / 'generative_model.ipynb'

print(f'reading  {nb_path}')
nb = nbformat.read(nb_path, as_version=4)

client = NotebookClient(
    nb,
    timeout=3600,
    kernel_name='p5-gen',
    resources={'metadata': {'path': str(here)}},
)
print('executing...')
client.execute()
nbformat.write(nb, nb_path)
print('done.')
