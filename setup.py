from setuptools import setup, find_packages

setup(
    name='netplay',
    version='0.1',
    author='Dominik Jeurissen',
    author_email='dominikjeurissen@web.de',
    description='A LLM-powered agent for playing NetHack.',
    packages=find_packages(),
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
)