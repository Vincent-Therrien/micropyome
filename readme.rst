micropyome
==========

- `English (en) <#Study-Microbiomes>`_
- `Français (fr) <#Étudier-les-microbiomes>`_


Soil Microbiome Predictions
---------------------------

This repository contains the data and code required to replicate our results
for soil microbiome predictions from environmental features.

Preliminary citation::

   Zahia Aouabed, Mohamed Achraf Bouaoune, Vincent Therrien, Mohammadreza Bakhtyari, Mohamed Hijri, and Vladimir Makarenkov. "Communities at Different Taxonomic Levels Using Machine Learning" (2024). Proceedings of the international conference IFCS-2024 (Costa Rica), Springer Verlag, pages 31-39.

A newer version of the research is to be published in a journal.


Installation
````````````

You can install the library by cloning the git repository and executing the
following command:

.. code-block:: bash

   pip install .


Results
```````

Our results are provided in Jupyter notebooks:

- The code required to replicate results with linear models is located in the
  notebook ``demo/regressions.ipynb``.
- The code required to replicate results with neural networks is located in the
  notebook ``demo/autoencoder.ipynb``.


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


Prédiction des microbiomes
--------------------------

Ce dépôt contient les données requises pour répliquer nos résultats de
prédiction d'abondances relatives d'organismes dans des microbiomes. L'article
suivant présente nos résultats préliminaires::

   Zahia Aouabed, Mohamed Achraf Bouaoune, Vincent Therrien, Mohammadreza Bakhtyari, Mohamed Hijri, and Vladimir Makarenkov. "Communities at Different Taxonomic Levels Using Machine Learning" (2024). Proceedings of the international conference IFCS-2024 (Costa Rica), Springer Verlag, pages 31-39.

Nous prévoyons publier Une version actualisée des résultats.
