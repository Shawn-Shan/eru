from setuptools import setup
from setuptools import find_packages

setup(name='eru',
      version='0.0.1',
      description='Deep Learning for all',
      author='Shawn Shan',
      author_email='shansixioing@uchicago.edu',
      url='https://github.com/Shawn-Shan/eru',
      license='MIT',
      install_requires=[],

      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
