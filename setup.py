from setuptools import setup, find_packages

setup(
      name='ravenclaw',
      version='0.1',
      description='For data wrangling.',
      url='https://github.com/IdinK/ravenclaw',
      author='Idin',
      author_email='d@idin.net',
      license='MIT',
      packages=find_packages(exclude=("jupyter_tests", "examples", ".idea", ".git")),
      install_requires=['numpy', 'pandas', 'SPARQLWrapper', 'slytherin', 'gobbledegook'],
      zip_safe=False
)
