#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(name='yumi_push',
      description='Multi-armed object singulation for ABB Yumi robot',
      version=0.1,
      url='https://github.com/simon-schaefer/yumi-push',
      author=['Simon Schaefer', 'Patrick Pfreundschuh'], 
      author_email='sischaef@ethz.ch',
      package_dir={'': 'src'},
      packages=find_packages('src'),

      install_requires=['gym==0.12.0',
	  					'numpy',
						'Pillow',
						'PyYAML==5.4',
						'scikit-image',
						'sklearn', 
						'opencv-python', 
						'matplotlib', 
						'pyinstrument',
						'seaborn',
						'tensorflow'
						],
      tests_require=[],

      # This avoids creating the egg file, which is a zip file, which makes our data
      # inaccessible by dir_from_package_name()
      zip_safe=False,

      # without this, the stuff is included but not installed
      include_package_data=True,

      classifiers=[
            # How mature is this project? Common values are
            #   2 - Prealpha
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 2 - Prealpha',

            # Specify the Python versions.
            'Programming Language :: Python :: 3',
      ],
)
