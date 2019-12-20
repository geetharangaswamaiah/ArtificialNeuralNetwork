from setuptools import setup

setup(
	name = 'ArtificialNeuralNetwork',
	version = '1.0',
	description = 'Python Library for Artificial Neural Network Algorithm',
	author = 'Geetha Rangaswamaiah',
	author_email = 'rgeetha2010@gmail.com',
	packages = ['ArtificialNeuralNetwork'],
	install_requires = [
		'pandas',
		'numpy',
		'scipy'
		],
	classifiers = [
		'License :: MIT License',
		'Programming Language :: Python :: 3.0',
		'Topic :: Recognition :: Artificial Neural Network'
		],
	include_package_data = True
)
