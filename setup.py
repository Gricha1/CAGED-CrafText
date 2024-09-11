from setuptools import setup, find_packages

setup(
    name='CrafText',
    version='0.1.0',
    description='A text processing package with various scenarios and checkers.',
    author='ZoyaV',
    url='https://github.com/ZoyaV/CrafText',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here, for example:
        # 'numpy>=1.18.5',
        # 'pandas>=1.1.3',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)