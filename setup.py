from setuptools import setup, find_packages
from os import path
import io

this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = [
        'langchain',
        'numpy',
        'pandas',
        'hana-ml>=2.24.25031800',
        'langchain-community',
        'langchain-core',
        'langchain-experimental',
        'langchain-text-splitters',
        'pydantic',
        'pydantic-core',
        'generative-ai-hub-sdk[all]'
]

setup(
    name='hana_ai',
    version="1.0.25032800",
    author='SAP',
    license='Apache License 2.0',
    url='https://github.com/SAP/generative-ai-toolkit-for-sap-hana-cloud',
    keywords='Generative AI Toolkit SAP HANA Cloud',
    description='Generative AI Toolkit for SAP HANA Cloud',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={'hana_ai': ['vectorstore/knowledge_base/sql_knowledge/*',
                              'vectorstore/knowledge_base/python_knowledge/*',
                              'include src/hana_ai/agents/scenario_knowledge_base/*']},
    include_package_data=True,
    install_requires=install_requires,
    python_requires='>=3.0'
)
