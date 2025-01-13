"""
Code templates.

This module contains tools for generating code templates.

The following functions are available:

    * :func `get_code_templates`
"""
# pylint: disable=consider-using-in
import os

def get_code_templates(option=None, customized_dir=None):
    """
    Get code templates.

    Parameters:
    -----------
    option: {'python', 'sql'}
        Option.

        Defaults to 'python'.
    customized_dir: str
        Customized directory.

        Defaults to None.
    """
    ids = []
    descriptions = []
    examples = []
    if option is None:
        option = 'python'
    if option:
        if option not in ['python', 'sql']:
            raise ValueError("option should be either 'python' or 'sql'")
    if option == 'python' or option == 'sql':
        temp_directory = os.path.join(os.path.dirname(__file__), "knowledge_base", "python_knowledge")
        if option == 'sql':
            temp_directory = os.path.join(os.path.dirname(__file__), "knowledge_base", "sql_knowledge")
        if customized_dir:
            temp_directory = customized_dir
        for filename in os.listdir(temp_directory):
            if filename.endswith('.txt'):
                with open(os.path.join(temp_directory, filename)) as f:
                    ids.append(filename.replace(".txt", ""))
                    contents = f.read()
                    #split contents by '------' into two parts: description and example
                    contents = contents.split('------', maxsplit=1)
                    #description
                    descriptions.append(contents[0])
                    #example
                    examples.append(contents[1])
    return {"id": ids, "description": descriptions, "example": examples}
