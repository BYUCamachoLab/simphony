import argparse
import os

parser = argparse.ArgumentParser(description='Extend the Simphony photonic simulator.')
parser.add_argument("--newelement", help="Start a new element.", type=str)
args = parser.parse_args()

name = args.newelement
os.mkdir(name)

contents_init = ""
contents_component = '''from simphony import components

# Create your component models here.
'''
contents_models = '''from simphony import components

# Create your simulation models here.
'''
contents_tests = '''

# Create your tests here.
'''

with open(os.path.join(name, "__init__.py"), "w+") as fout:
    fout.write(contents_init)

with open(os.path.join(name, "component.py"), "w+") as fout:
    fout.write(contents_component)

with open(os.path.join(name, "models.py"), "w+") as fout:
    fout.write(contents_models)

with open(os.path.join(name, "tests.py"), "w+") as fout:
    fout.write(contents_tests)