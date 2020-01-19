from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="shor",  
    version="0.1",  
    description="Educational Python walkthrough of Shor's quantum algorithm for factorising numbers",  
    long_description=long_description,  
    long_description_content_type="text/markdown",
    author="timhunderwood",  
    keywords="quantum physics educational",  
    package_dir={"": "shor"},  
    packages=find_packages(where="shor"),
    python_requires=">=3.7",
    install_requires=["numpy", "matplotlib"],  
    project_urls={  
        "Blog post": "https://nonetype.eu.pythonanywhere.com/articles/0008"
    },
)
