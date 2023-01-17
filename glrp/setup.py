#!/usr/bin/env python

import os
from setuptools import setup, find_packages

# read requirements
with open('requirements.txt') as file:
    rq = file.read().splitlines()

setup(name='GLRP_GCNN',
      version='0.1',
      description='glrp implementation with tf2 and spektral',
      packages=find_packages(include=['glrp', 'glrp.*']),
      install_requires=rq
)
