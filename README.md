#  Mass Spectrum Predictor

This repository contains implementations for the following spectrum simulator models predicting molecular tandem mass spectra from molecules: 

<!-- TODO: Add link to ICEBERG paper-->
- ️️️️ ❄️ ICEBER️️G ❄️: [Inferring CID by Estimating Breakage Events and Reconstructing their Graphs]()
- 🧣 SCARF 🧣: [Subformula Classification for Autoregressively Reconstructing Fragmentations](https://arxiv.org/abs/2303.06470),

ICEBERG predicts spectra at the level of molecular fragments, whereas SCARF predicts spectra at the level of chemical formula. Additional sensible baselines using neural networks to predict "binned" spectra are also included, along with CFM-ID instructions.

Contributors: Sam Goldman, John Bradshaw, Janet Li, Jiayi Xin, Connor W. Coley


![Model graphic](github_teaser.png)

## Contents


1. [Install](#setup)  
2. [Quickstart](#quickstart)  
3. [Data](#data)  
4. [Experiments](#experiments)    
5. [Analysis](#analysis)    


## Install & setup <a name="setup"></a>

Install and set up the conda environment using [mamba](https://mamba.readthedocs.io/en/latest/):

```
mamba env create -f environment.yml
mamba activate ms-gen
pip install -r requirements.txt
python3 setup.py develop
```


## Quickstart <a name="quickstart"></a>

To make predictions, we have released and made public a version of SCARF and ICEBERG both trained upon the CANOPUS dataset (renamed NPLIB1 in the ICEBERG manuscript). This can be downloaded and used to predict a set of 100 sample molecules contained in the NIST library, as included at `data/spec_datasets/sample_labels.tsv` (test set):

 
```
# Scarf quickstart
. quickstart/scarf/download_model.sh
. quickstart/scarf/run_model.sh

# Iceberg quickstart
. quickstart/iceberg/download_model.sh
. quickstart/iceberg/run_model.sh
```

Model outputs will be contained in `quickstart/{model}/out/`. **We note that this model may be less performant than the model trained on the commercial NIST20 Library. Download links to models trained on NIST20 models are available upon request to any users with a NIST license.**  ICEBERG assigns intensities to various fragment masses. Refer to `notebooks/iceberg_demo.ipynb` for a walkthrough for how to process and interpret the output dictionary.


## Data <a name="data"></a>

A data subset from the GNPS database can first be downloaded and older sirius
outputs and magma runs removed.

```
. data_scripts/download_gnps.sh
```

Structural data splits can be established with `data_scripts/make_splits.py`, and the following command will create all splits in bulk:

```
. data_scripts/all_create_splits.sh
```

### SCARF Processing

Data should then be assigned to subformulae files using
`data_scripts/forms/assign_subformulae.py`, which will preprocess the data. We
produce two fragment versions of the molecule, `magma_subform_50` and
`no_subform`. The former strictly labels subformula based upon smiles structure
and the latter is permissive and allows all entries to pass. The following
script does this for both `canopus_train_public` and `nist20` if it has been
acquired and parsed (commercial dataset).

```
. data_scripts/assign_subform.sh
```


### ICEBERG Processing

In addition to building processed subformulae, to train ICEBERG, we must
annotate substructures and create a labeled dataset over the breakage process, 
which we do with the MAGMa algorithm.

This can be done with the following script, specifying an appropriate dataset:

```
. data_scripts/dag/run_magma.sh
```


### Retrieval

To conduct retrieval experiments, libraries of smiles must be created. A PubChem
library is converted and each chemical formula is mapped to (smiles, inchikey)
pairs. Subsets are selected for evaluation.  Making formula subsets takes longer
(on the order of several hours, even parallelized) as it requires converting
each molecule in pubchem to a mol / InChi. 

```
source data_scripts/pubchem/01_download_smiles.sh
python data_scripts/pubchem/02_make_formula_subsets.py
python data_scripts/pubchem/03_make_retrieval_lists.py
```

To quickly download tables for canopus:  

```
cd data/spec_datasets/canopus_train_public/
wget https://www.dropbox.com/s/7zr6euhuhz3ohi9/retrieval_tbls.tar
tar -xvf canopus_train_public.tar
```

## Experiments <a name="experiments"></a>


### SCARF

SCARF models trained in two parts: a prefix tree generator and an intensity predictor. The pipeline for training and evaluating this model can be accessed in `run_scripts/scarf_model/`. The internal pipeline used to conduct experiments can be followed below:

1. *Hyperopt scarf model*: `run_scripts/scarf_model/01_hyperopt_scarf.sh`
2. *Train scarf model*: `run_scripts/scarf_model/02_run_scarf_gen_train.sh`
3. *Sweep number of prefixes to generate*: `run_scripts/scarf_model/03_sweep_scarf_gen_thresh.py`  
4. *Use model 1 to predict model 2 training set*: `run_scripts/scarf_model/04_scarf_gen_predict.sh`   
5. *Add intensity targets to predictions*: `run_scripts/scarf_model/05_process_scarf_train.py`
6. *Hyperopt scarf inten model*: `run_scripts/scarf_model/06_hyperopt_scarf_inten.sh`
7. *Train intensity model*: `run_scripts/scarf_model/07_train_scarf_inten.sh`
8. *Make and evaluate intensity predictions*: `run_scripts/scarf_model/08_predict_form_inten.py`
9. *Run retrieval*: `run_scripts/scarf_model/09_run_retrieval.py`  
10. *Time scarf*: `run_scripts/scarf_model/10_time_scarf.py`  
11. *Export scarf forms* `run_scripts/scarf_model/11_export_forms.py`


Instead of running in batched pipeline model, individual gen training, inten
training, and predict calls can be  made using the following scripts respectively:

1. `python src/ms_pred/scarf_pred/train_gen.py`
2.  `python src/ms_pred/scarf_pred/train_inten.py`
3.  `python src/ms_pred/scarf_pred/predict_smis.py`


### ICEBERG

ICEBRG models, like SCARF, are trained in two parts: a learned fragment generator and an intensity predictor. The pipeline for training and evaluating this model can be accessed in `run_scripts/dag_model/`. The internal pipeline used to conduct experiments can be followed below:

1. *Hyperopt dag model*: `run_scripts/dag_model/01_hyperopt_dag.sh`   
2. *Train dag model*: `run_scripts/dag_model/02_run_dag_gen_train.sh`   
3. *Sweep over the number of fragments to generate*: `run_scripts/dag_model/03_sweep_gen_thresh.py`    
4. *Use model 1 to predict model 2 training set*: `run_scripts/dag_model/04_run_dag_gen_predict.sh`   
5. *Add intensity targets to predictions*: `run_scripts/dag_model/05_process_dag_train.py`
6. *Hyperopt dag inten model*: `run_scripts/dag_model/06_hyperopt_inten.sh`
7. *Train intensity model*: `run_scripts/dag_model/07_train_dag_inten.sh`
8. *Make and evaluate intensity predictions*: `run_scripts/dag_model/08_predict_dag_inten.py`
9. *Run retrieval*: `run_scripts/dag_model/09_run_retrieval.py`  
10. *Time iceberg*: `run_scripts/dag_model/10_time_dag.py`  
11. *Export dag predictions* `run_scripts/dag_model/11_export_preds.py`


Instead of running in batched pipeline model, individual gen training, inten
training, and predict calls can be  made using the following scripts respectively:

1. `python src/ms_pred/dag_pred/train_gen.py`
2.  `python src/ms_pred/dag_pred/train_inten.py`
3.  `python src/ms_pred/dag_pred/predict_smis.py`

An additional notebook showcasing how to individually load models and make predictions can be found at `notebooks/iceberg_fig_qualitative.ipynb`.

### FFN Spec 

Experiment pipeline utilized:  
1. *Hyperopt model*: `run_scripts/ffn_model/01_hyperopt_ffn.sh`
2. *Train models*: `run_scripts/ffn_model/02_run_ffn_train.sh`
3. *Predict and eval*: `run_scripts/ffn_model/03_predict_ffn.py`
4. *Retreival experiments*: `run_scripts/ffn_model/04_run_retrieval.py`
5. *Time ffn*: `run_scripts/ffn_model/05_time_ffn.py`


### GNN Spec 

Experiment pipeline:   
1. *Hyperopt model*: `run_scripts/gnn_model/01_hyperopt_ffn.sh`
2. *Train models*: `run_scripts/gnn_model/02_run_ffn_train.sh`
3. *Predict and eval*: `run_scripts/gnn_model/03_predict_ffn.py`
4. *Retreival experiments*: `run_scripts/gnn_model/04_run_retrieval.py`
5. *Time ffn*: `run_scripts/gnn_model/05_time_gnn.py`

### CFM-ID

CFM-ID is a well-established fragmentation-based mass spectra prediction model. We include brief instructions for utilizing this tool below

Build docker: 
```
docker pull wishartlab/cfmid:latest
```

Make prediction:
```
. run_scripts/cfm_id/run_cfm_id.py
. run_scripts/cfm_id/process_cfm.py
. run_scripts/cfm_id/process_cfm_pred.py
```


### Freq baselines

As an addiitonal baseline to compare to the generative portion of our scarf
(thread), we include frequency baselines for generating form subsets:

```
. run_scripts/freq_baseline/predict_freq.py
. run_scripts/freq_baseline/predict_rand.py
```


## Analysis <a name="analysis"></a>

Analysis scripts can be found in `analysis` for evaluating both formula
predictios `analysis/form_pred_eval.py` and spectra predictions
`analysis/spec_pred_eval.py`.

Additional analyses used for figure generation were conducted in `notebooks/`.


## Citation

We ask any user of this repository to cite the following works based upon the portion of the repository used:
<!-- TODO: Add citation for  ICEBERG -->
```
@article{https://doi.org/10.48550/arxiv.2303.06470,
  doi = {10.48550/ARXIV.2303.06470},
  
  url = {https://arxiv.org/abs/2303.06470},
  
  author = {Goldman, Samuel and Bradshaw, John and Xin, Jiayi and Coley, Connor W.},
  
  keywords = {Quantitative Methods (q-bio.QM), Machine Learning (cs.LG), FOS: Biological sciences, FOS: Biological sciences, FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Prefix-tree Decoding for Predicting Mass Spectra from Molecules},
  
  publisher = {arXiv},
  
  year = {2023},
  
  copyright = {Creative Commons Attribution Share Alike 4.0 International}
}

@article{goldman2023iceberg,
  doi = {TODO}
  url = {TODO},
  
  author = {Goldman, Samuel and Li, Janet and Coley, Connor W.},
  
  keywords = {},
  
  title = {},
  
  publisher = {arXiv},
  
  year = {2023},
  
  copyright = {Creative Commons Attribution Share Alike 4.0 International}
} 


```

In addition, we utilize both the NEIMS approach for our binned FFN and GNN
baselines, MAGMa for constructing formula labels, and CFM-ID as a baseline. We
encourage the following additional citations:

1.   Wei, Jennifer N., et al. "Rapid prediction of electron–ionization mass spectrometry using neural networks." ACS central science 5.4 (2019): 700-708.
2. Ridder, Lars, Justin JJ van der Hooft, and Stefan Verhoeven. "Automatic compound annotation from mass spectrometry data using MAGMa." Mass Spectrometry 3.Special_Issue_2 (2014): S0033-S0033.
3. Wang, Fei, et al. "CFM-ID 4.0: more accurate ESI-MS/MS spectral prediction and compound identification." Analytical chemistry 93.34 (2021): 11692-11700.
