import papermill as pm
import os
import glob
import pytest


@pytest.mark.parametrize("notebook_path", glob.glob(os.path.join(os.path.dirname(__file__), "*.ipynb")))
def test_notebook_runner(notebook_path):
    """
    This pytest function runs all the ipynb files in the repository
    using papermill - a notebook runner that throws if any exception occurs while executing the notebook
    """
    # Run all the notebooks in the repository
    pm.execute_notebook(notebook_path, "-")
