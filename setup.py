from setuptools import setup, find_packages

def get_version():
    """ fancy doc to satisfy linter """
    with open('version.txt') as ver_file:
        version_str = ver_file.readline().rstrip()
    return version_str

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='hana_ai',
    version=get_version(),
    author='SAP',
    license='Apache License 2.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={'hana_ai': ['vectorstore/knowledge_base/sql_knowledge/*',
                              'vectorstore/knowledge_base/python_knowledge/*',
                              'include src/hana_ai/agents/scenario_knowledge_base/*']},
    install_requires=required,
    include_package_data=True,
    python_requires='>=3.0'
)
