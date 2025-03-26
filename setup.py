from setuptools import setup, find_packages

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
        'generative-ai-hub-sdk'
]

setup(
    name='hana_ai',
    version="1.0.25031900",
    author='SAP',
    license='Apache License 2.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={'hana_ai': ['vectorstore/knowledge_base/sql_knowledge/*',
                              'vectorstore/knowledge_base/python_knowledge/*',
                              'include src/hana_ai/agents/scenario_knowledge_base/*']},
    include_package_data=True,
    install_requires=install_requires,
    python_requires='>=3.0'
)
