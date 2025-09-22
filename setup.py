from setuptools import setup, find_packages

setup(
    name='LLM-Local-Host',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    description='Code to locally host a HF LLM on a GPU',
    author='Sharvil D',
    author_email='sharvil.public@gmail.com',
    url='',
    install_requires=[
        "torch==2.6.0+cu124", # Set CUDA according to the local system reqs
        "transformers==4.56.2",
        "accelerate==1.10.1",
        "bitsandbytes==0.47.0",
        "pydantic==2.11.9",
        "pydantic_core==2.33.2",
        "fastapi==0.116.2",
        "uvicorn==0.36.0"
    ]
)