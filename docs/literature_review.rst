Literature Review
=================

.. bibliography:: references.bib
   :filter: False


Microbiome Crash Course for Folks without an Ecology Degree!
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A microbiome is a "community of microorganisms (such as fungi, bacteria and
viruses) that exists in a particular environment" :cite:`nhgri`.

Ecosystems can be studied and sampled at different scales. In the context of
soil microbiomes, three scales often show up :cite:`neon_definition`:

- A **site** is the area that is studied, such as a field or a pond.
- A **plot** is a smaller area within a site chosen for a particular set of
  conditions, such as the presence of vegetation.
- A **core** is a soil sample collected within a plot for analysis.


Lexicon:

+----------------------+------------------------------------------------------+
| Word                 | Definition                                           |
+======================+======================================================+
| Edaphic              | Resulting from or influenced by the soil rather than |
|                      | the climate (Merriam-Webster)                        |
+----------------------+------------------------------------------------------+
| ESV                  | Exact Sequence Variant :cite:`averill2021`           |
+----------------------+------------------------------------------------------+
| ITS                  | Internal Transcribed Spacer, the DNA segment between |
|                      | two rRNA subunits                                    |
+----------------------+------------------------------------------------------+
| OTU                  | Operational Taxonomic Unit :cite:`averill2021`       |
+----------------------+------------------------------------------------------+
| PD                   | Phylogenetic diversity                               |
+----------------------+------------------------------------------------------+
| Rapoport's rule      | Species’ latitudinal ranges increase toward the poles|
|                      | and diversity increases toward the equator           |
|                      | :cite:`tedersoo2014`                                 |
+----------------------+------------------------------------------------------+


Soil Microbiome Predictions
+++++++++++++++++++++++++++

Averill et al :cite:`averill2021` investigated whether the composition of soil
microbiome can be predicted before observation. The following list summarizes
the key points of their research:

- The **input data** are soil conditions, such as temperature, pH, or rainfall,
  as well are the spatial location of the microbiome.
- The **output data** are taxonomic groups of the microorganisms that make up
  the soil microbiome, which are analyzed at the following levels: functional
  (e.g. nitrogen-fixating, mycorrhizal fungi, ...), phylum, class, order,
  family, and genus.
- The machine learning model is a Dirichlet multivariate regression model that
  predicts functional and taxonomic groups.
- They used the Moran I score to evaluate the spatial autocorrelation of
  organisms at different taxonomic scales. The more general a group is, the
  higher is the Moran I score (i.e. they seem less "patchy" when viewed from
  above).

Averill et al used the following datasets:

Fungi:

- Tedersoo et al :cite:`tedersoo2014`

Bacteria:

- Bahram et al :cite:`bahram2018`
- Delgado-Baquerizo et al :cite:`delgadobaquerizo2018`
- Ramirez et al :cite:`ramirez2018` (only for calibration)


Microbiome Datasets
+++++++++++++++++++


General Datasets
----------------

The Earth Microbiome Project :cite:`emp` contains samples from soil and marine
ecosystems.


Bacteria Datasets
-----------------


1. Bahram2018
`````````````

Bahram et al :cite:`bahram2018` used 7560 samples collected across 189 sites to
demonstrate that "bacterial, but not fungal, genetic diversity is highest in
temperate habitats and that microbial gene composition varies more strongly
with environmental variables than with geographic distance". Their analysis is
based on metagenomics and metabarcoding.

Raw data:

+--------------+---------------------------+--------------------------------------------------+
| SRA  Number  | Description               |     Link                                         |
+==============+===========================+==================================================+
| PRJEB24121   | Estonian forest and       | https://www.ncbi.nlm.nih.gov/sra?term=PRJEB24121 |
|              | grassland topsoil samples |                                                  |
+--------------+---------------------------+--------------------------------------------------+
| PRJEB19856   | 16S metabarcoding data of | https://www.ncbi.nlm.nih.gov/sra?term=PRJEB19856 |
|              | global soil samples       |                                                  |
+--------------+---------------------------+--------------------------------------------------+
| PRJEB19855   | 18S metabarcoding data of | https://www.ncbi.nlm.nih.gov/sra?term=PRJEB19855 |
|              | global soil samples       |                                                  |
+--------------+---------------------------+--------------------------------------------------+
| PRJEB18701   | Global analysis of soil   | https://www.ncbi.nlm.nih.gov/sra?term=PRJEB18701 |
|              | microbiomes               |                                                  |
+--------------+---------------------------+--------------------------------------------------+


2. Delgado-Baquerizo2018
````````````````````````

Delgado-Baquerizo et al :cite:`delgadobaquerizo2018` have compiled a global
atlas of bacteria found in soil at 237 locations across 6 continents. Soil
bacteria are not well studied because they are hard to cultivate in laboratories
and genetic information is available for few of them. They determined that plant
productivity is the best bacterial distribution predictor. The raw FASTQ
sequencing files are publicly available.

Raw data: https://figshare.com/s/82a2d3f5d38ace925492


3. Ramirez2018
``````````````

Ramirez et al :cite:`ramirez2018` merged 30 independent bacterial datasets
comprising 1998 samples with machine a random forest model. Organisms are
classified based on their operational taxonomic units (OTU), which are
determined from DNA sequence similarity. Their dataset **includes** sequences
obtained with Roche 454 sequencing that Averill et al :cite:`averill2021`
ignored to avoid bias. The raw sequences and processed data are available.

Processed data: https://www.nature.com/articles/s41564-017-0062-x#Sec14


Fungi Datasets
--------------


1. Tedersoo2014
```````````````

Tedersoo et al :cite:`tedersoo2014` aggregated data from 365 sites to
characterize the world distribution of fungi. They found that distance from
the equator and mean annual precipitation had the strongest effect on richness
of fungi.

Raw data: https://www.ncbi.nlm.nih.gov/sra?term=SRP043706


References
++++++++++

.. bibliography:: references.bib
