# Mitochondrial toxicity prediction
Supporting information from "Predicting the Mitochondrial Toxicity of Small Molecules: Insights From Mechanistic Assays and Cell Painting Data" by Marina Garcia de Lomana, Paula Andrea Marin Zapata, and Floriane Montanari (<i>Chem. Res. Toxicol.</i>, 2023)\
https://doi.org/10.1021/acs.chemrestox.3c00086

## Abstract
Mitochondrial toxicity is a significant concern in the drug discovery process, as compounds that disrupt the function of these organelles can lead to serious side effects, including liver injury and cardiotoxicity. Different in vitro assays exist to detect mitochondrial toxicity at varying mechanistic levels: disruption of the respiratory chain, disruption of the membrane potential, or general mitochondrial dysfunction. In parallel, whole cell imaging assays like Cell Painting provide a phenotypic overview of the cellular system upon treatment and enable the assessment of mitochondrial health from cell profiling features. In this study, we aim to establish machine learning models for the prediction of mitochondrial toxicity, making the best use of the available data. For this purpose, we first derived highly curated datasets of mitochondrial toxicity, including subsets for different mechanisms of action. Due to the limited amount of labeled data often associated with toxicological endpoints, we investigated the potential of using morphological features from a large Cell Painting screen to label additional compounds and enrich our dataset. Our results suggest that models incorporating morphological profiles perform better in predicting mitochondrial toxicity than those trained on chemical structures alone (up to +0.08 and +0.09 mean MCC in random and cluster cross-validation, respectively). Toxicity labels derived from Cell Painting images improved the predictions on an external test set up to +0.08 MCC. However, we also found that further research is needed to improve the reliability of Cell Painting image labeling. Overall, our study provides insights into the importance of considering different mechanisms of action when predicting a complex endpoint like mitochondrial disruption as well as into the challenges and opportunities of using Cell Painting data for toxicity prediction.

## Repository information
The training data for the mechanistic models, as well as the external test set can be downloaded from the supporting information of the paper. 

The best model for each endpoint (<i>"overall"</i>, <i>"membrane potential"</i>, <i>"respiratory chain"</i> and <i>"function of mitochondria"</i>) is provided under <b>models/</b>. The models for the <i>"overall"</i> and <i>"membrane potential"</i> endpoints are feed-forward neural networks and the models for the <i>"respiratory chain"</i> and <i>"function of mitochondria"</i> endpoints are random forest models. 

An example on how to run the inference code, as well as how to prepare the input SMILES, can be found in the Jupyter Notebook (<b>apply_models.ipynb</b>).


## Dependencies
- python 
- rdkit 2021.09.4
- scikit-learn 1.0.2
- pytorch 1.10.2

An evironment fulfilling these dependencies can be created by cloning the repository and running:

<code>conda env create -f mtx_env.yaml</code>


