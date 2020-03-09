from setuptools import setup, find_packages

setup(
	name='ravenclaw',
	version='2020.1.6',
	description='For data wrangling.',
	url='https://github.com/idin/ravenclaw',
	author='Idin',
	author_email='py@idin.ca',
	license='MIT',
	packages=find_packages(exclude=("jupyter_tests", "examples", ".idea", ".git")),
	install_requires=[
		'numpy', 'pandas', 'chronometry', 'SPARQLWrapper', 'slytherin', 'fuzzywuzzy', 'joblib'
	],
	zip_safe=False
)
