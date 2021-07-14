from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(name='emissions',
      version='0.1',
      description='predicts the result of vehicle emissions test',
      url='https://github.com/Guli-Y/wimlds_emissions',
      author='wimlds-Berlin-team',
      author_email='',
      license='',
      packages=['emissions'],
      install_requires=requirements,
      zip_safe=False)