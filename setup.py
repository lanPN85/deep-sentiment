from setuptools import setup

setup(
    name='deep-sentiment',
    version='0.1.0',
    packages=['sentiment'],
    url='https://github.com/lanPN85/deep-sentiment',
    license='MIT',
    author='Phan Ngoc Lan, Nguyen Duy Manh, Thieu Hai Hoan',
    author_email='phan.ngoclan58@gmail.com',
    description='LSTM-CNN sentiment analysis library',
    install_requires=[
        'keras==2.0.6', 'tensorflow>=1.2.1',
        'gensim==2.2.0', 'nltk>=3.2.4', 'h5py>=2.7.0'
    ]
)
