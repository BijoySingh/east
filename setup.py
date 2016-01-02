"""
Emotion and Sentiment Library for Python and Command Line
"""
from setuptools import find_packages, setup

dependencies = ['nltk', 'scipy', 'numpy', 'scikit-learn', 'progressbar']

setup(
    name='east',
    version='1.0.0',
    url='https://github.com/BijoySingh/east',
    license='BSD',
    author='Bijoy Singh Kochar',
    author_email='bijoysingh693@gmail.com',
    description='Emotion and Sentiment Library for Python and Command Line',
    long_description=__doc__,
    packages=['east'],
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    install_requires=dependencies,
    entry_points={
        'console_scripts': [
            'east = east.cli:main',
        ],
    },
    classifiers=[
        # As from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        # 'Development Status :: 1 - Planning',
        # 'Development Status :: 2 - Pre-Alpha',
        # 'Development Status :: 3 - Alpha',
        # 'Development Status :: 4 - Beta',
        'Development Status :: 5 - Production/Stable',
        # 'Development Status :: 6 - Mature',
        # 'Development Status :: 7 - Inactive',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)