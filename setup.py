from setuptools import setup, find_packages

setup(
	name='ravenclaw',
	version='1.3.4',
	description='For data wrangling.',
	url='https://github.com/idin/ravenclaw',
	author='Idin',
	author_email='py@idin.ca',
	license='MIT',
	packages=find_packages(exclude=("jupyter_tests", "examples", ".idea", ".git")),
	install_requires=['numpy', 'pandas', 'chronology', 'SPARQLWrapper', 'slytherin'],
	zip_safe=False
)
