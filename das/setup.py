from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'scipy',
    'scikit-learn==0.20.0',
    'joblib',
    'xgboost',
    'lightgbm',
    'psutil',
    'ConfigSpace==0.4.7',
    'statsmodels==0.8.0',
    'matplotlib'
]

setup(
    name="das",
    version="1.0",
    include_package_data=True,
    author="DAS Contributors",
    author_email="xx@xx.com",
    url="xx",
    packages=find_packages(),
    install_requires=install_requires,
    description="DAS Framework",
    platforms='python 3.6',
)
