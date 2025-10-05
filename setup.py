from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]: 
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]  # while reading line by line "\n" will added so we have to remove it

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
name='ml_project',
version='0.0.1',
author='Ashish',
author_email='ashishofficial12321@gmail.com',
packages = find_packages(),
install_requirements = get_requirements('requirements.txt')

)