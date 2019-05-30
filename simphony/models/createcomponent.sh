#! /bin/bash
#
# Author: Sequoia Ploeg
#
# Purpose: Initialize the folders and files required for a new component
# module. Automatically writes __init__.py and also provides a stubbed
# class with functions for the component.

echo "Component Creation Wizard"
echo "-------------------------"
echo "Welcome!"
echo "Type the new component's name, followed by [ENTER]: "

read module

if [ ! -d $module ]
then
    mkdir $module
else
    echo "'$module' already exists!"
    exit 1
fi

cd $module

touch __init__.py
echo "from .model import Model" >> __init__.py

touch model.py
contents="

class Model:
    def __init__(self):
        pass

    # TODO: Provide the arguments required by the model and implement get_s_params
    @staticmethod
    def get_s_params(*args, **kwargs):
        pass

    @staticmethod
    def about():
        message = \"About $module:\"
        print(message)
"
echo "$contents" >> model.py

echo "Please add '$module' to INSTALLED_COMPONENTS in 'components.py' 
(also include it in whichever component model it belongs to)."