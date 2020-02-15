from setuptools import setup
import io

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name='dbstep',
  packages=['dbstep'],
  version='1.0',
  description='Dft Based Steric Parameters',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author='',
  author_email='robert.paton@colostate.edu',
  keywords=['compchem', 'steric','sterimol','informatics'],
  classifiers=[],
  install_requires=["numpy", ],
  python_requires='>=2.6',
  include_package_data=True,
)
