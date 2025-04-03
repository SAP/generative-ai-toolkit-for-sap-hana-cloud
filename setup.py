from setuptools import setup, find_packages
from os import path
import io

this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

def parse_requirements(filename):
    with open(filename, 'r') as req_file:
        return [line.strip() for line in req_file if line.strip() and not line.startswith('#')]

install_requires = parse_requirements('requirements.txt')

def get_version():
    with open('version.txt') as ver_file:
        version_str = ver_file.readline().rstrip()
    return version_str

setup(
    name="hana_ai",
    version="1.0.25040300",
    author='SAP',
    license='Apache License 2.0',
    url='https://github.com/SAP/generative-ai-toolkit-for-sap-hana-cloud',
    project_urls={
    'Documentation': 'https://sap.github.io/generative-ai-toolkit-for-sap-hana-cloud/',
    'Report Issues': 'https://github.com/SAP/generative-ai-toolkit-for-sap-hana-cloud/issues',
    'Notebook Examples': 'https://github.com/SAP/generative-ai-toolkit-for-sap-hana-cloud/tree/main/nutest/testscripts'
    },
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
