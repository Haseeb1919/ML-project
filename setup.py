#require packages
from setuptools import setup, find_packages, setup
from typing import List


HYPHEN_E_DOT = "-e ."

# function to get requirments in requirments.txt
def get_requirements(file_path: str)-> List[str]:
    #read requrirments from file
    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        #remove \n from requirments with list comprehension
        requirements = [i.replace("\n"," ") for i in requirements]

        #remove HYPHEN_E_DOT from requirments.txt
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

   


setup(

    name="mlproject",
    version="0.0,1",
    author="haseeb",
    author_email="haseeburrehman1919@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt') # use function to get requirments in requrirments.txt


)
