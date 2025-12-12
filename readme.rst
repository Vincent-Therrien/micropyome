Soil Microbiome Predictions
===========================

This repository contains the data and code required to replicate our results
for soil microbiome predictions from environmental features.


Installation
````````````

You can install the library by cloning the git repository and executing the
following command:

.. code-block:: bash

   pip install .

The code required to replicate results with linear models is located in the notebook
``demo/regressions.ipynb``.


Project Organization
````````````````````

The project is organized as follows:

- **micropyome**: Python source code of the library.
- **docs**: Sphinx source files for the documentation. You can build the
  documentation by executing the command ``make html`` inside of the ``docs``
  directory.
- **demos**: Jupyter notebooks that show how to use the library.
- **tests**: Test suite of the project, written with Pytest. You can run the
  tests by executing the command ``pytest tests``.


Runnable example
````````````````

This section provides a small example with step-by-step instructions (notebook) and environment file
(``requirements.txt``) necessary to replicate our results.

- The **input data** are located in the directory ``data``. It comprises the following subdirectories:

  - ``data/averill/bacteria`` contains the following files for multiple taxonomic levels:

    - ``05_variables.csv`` contains 5 environmental variables for observed data when collecting samples.
    - ``15_variables.csv`` contains 15 environmental variables for observed data when collecting samples.
    - ``observed.csv`` contains observed abundances of bacteria.
    - ``predicted.csv`` contains abundances of bacteria predicted by the model of Averill et al.
    - ``y_11groupTaxo.csv`` contains observed abundances of the 11 most abundant bacterial taxa.

To load the data and train classifier to replicate our results, you need to:

1. Create a Python virtual environment to install all dependencies. You can create a virtual environment with the command ``python3 -m venv <name>`` on Linux or ``py -m venv <name>`` on Windows, where ``<name>`` is an arbitrary name given to the virtual environment.
2. Activate the virtual environment. In Linux, run the command ``source <name>/Scripts/activate``. In Windows, run the command ``<name>\Scripts\activate``.
3. Install dependencies: ``pip install .``. This will install all external dependencies and project-specific code developed for this project.
4. You can now open the notebook ``demon/regression.ipynb`` and execute it with the Python virtual environment that you configured. The notebook cells filter and normalize input data, train machine learning models for regression, and evaluates the trained models using R^2.
5. For example, using the input files ``15_variables.csv``, ``observed.csv``, ``y_11groupTaxo.csv``, and ``y_11groupTaxo.csv``, from the directory ``data/averill/bacteria``, we obtain the results shown in the article for the bacterial dataset of Averill et al. - they will be displayed in the notebook.


Reference:
``````````

Zahia Aouabed, Vincent Therrien, Mohamed Achraf Bouaoune, Mohammadreza Bakhtyari, Mohamed Hijri, and Vladimir Makarenkov. "Soil microbiome prediction using traditional machine learning and deep learning models" (2025), submitted to Scientific Reports.

