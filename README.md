# RAB-Cred

This is the official code repository for the paper: **LLMs as annotators of credibility assessment in Danish asylum decisions: evaluating classification performance and errors beyond aggregated metrics**, to be presented at the 20th Linguistic Annotation Workshop (LAW-XX) @ ACL 2026.

## Table of Contents

- [Set-up](#set-up)
- [RAB-Cred Dataset](#rab-cred-dataset)
- [Extract our LLM annotations](#extract-the-llm-annotations)
- [Reproducing the LLM annotations](#reproducing-the-llm-annotations)
- [Reproducing the analysis and figures](#reproducing-the-analysis-and-figures)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)


## Set-up

The code was tested on Ubuntu 24.04 with Python 3.12, torch==2.10.0 and transformers==5.0.0. We recommend using uv to set up the environment, but you can also use pip if you prefer.

Clone the repository:
```bash
git clone https://github.com/glhr/RAB-Cred
cd RAB-Cred
```

Install requirements using uv (recommended):
```bash
wget -qO- https://astral.sh/uv/install.sh | sh # install uv, see https://docs.astral.sh/uv/getting-started/installation/#installation-methods
uv sync . # install dependencies
```

Alternatively, install requirements using pip:
```bash
# 1. Create a virtual environment (optional but recommended):
python -m venv .venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## RAB-Cred Dataset

The RAB-Cred dataset is available on HuggingFace: [https://huggingface.co/datasets/XAI-CRED/RAB-Cred](https://huggingface.co/datasets/XAI-CRED/RAB-Cred). You will need to request access before being able to download the dataset - we will review access requests as soon as possible.

Once you have access, download the dataset at the root of this repo using git or the HuggingFace CLI:

```bash
## Make sure git-xet is installed (https://hf.co/docs/hub/git-xet)
curl -sSfL https://hf.co/git-xet/install.sh | sh
git clone https://huggingface.co/datasets/XAI-CRED/RAB-Cred

# Or you can use the huggingface CLI:
## Make sure the hf CLI is installed
curl -LsSf https://hf.co/cli/install.sh | bash
hf datasets download XAI-CRED/RAB-Cred
```

## Extract our LLM annotations

The LLM annotations generated in our paper for the validation set (21 models x 30 prompt combinations = 630 distinct LLM annotators) and the test set (5 models x 3 prompt combinations = 15 distinct LLM annotators) are available in the zip file [experiment_results.zip](experiment_results.zip). In the root of this repo, extract the LLM annotations for the validation and test sets as follows:

```bash
unzip experiment_results.zip -d experiment_results
```

The structure of the repo should now be as follows:
```bash
.
├── analysis # contains the code for the analysis and figures in the paper
├── experiment_configs # config files for generating LLM annotations
├── experiment_results # our results used in the paper
│   ├── experiment_results_test200
│   └── experiment_results_val
├── llm_annotation # code for generating LLM annotations
└── RAB-Cred # the downloaded RAB-Cred dataset from HuggingFace
```

## Reproducing the LLM annotations

To generate annotations for a given subset (validation or test) and a given model-prompt combination, run the following command:
```bash
subset=val # or test200
HF_TOKEN=your_huggingface_token_here python3 llm_annotation/get_llm_preds.py config_path=experiment_configs/ACLW_${subset}$.yml
```
This will load the default configuration defined in [experiment_configs/ACLW_val.yml](experiment_configs/ACLW_val.yml) or [experiment_configs/ACLW_test200.yml](experiment_configs/ACLW_test200.yml), which you can modify to specify the model-prompt combination of interest. Note:
* the model name is the huggingface name of the model to use (e.g. "ibm-granite/granite-4.0-micro")
* the available prompts are in [llm_annotation/prompts_system](llm_annotation/prompts_system) and [llm_annotation/prompts_user](llm_annotation/prompts_user)
* the generated LLM annotations will be saved in the folder specified in the config file.
* note: the HF_TOKEN is only required for gated HuggingFace models (e.g. Gemma3).

## Reproducing the analysis and figures

Note: double-check the paths defined in [analysis/paths.py](analysis/paths.py) to make sure they match the paths in your local set-up.

Data exploration: [analysis/data_exploration.ipynb](analysis/data_exploration.ipynb)

Validation set evaluation: [analysis/val_results.ipynb](analysis/val_results.ipynb)

Test set evaluation: [analysis/test_results.ipynb](analysis/test_results.ipynb)

Prompt and model sensitivity: [analysis/test_prompt_sensitivity.ipynb](analysis/test_prompt_sensitivity.ipynb)

Credibility assessment vs. outcome correlation:
[analysis/val_outcome_correlation.ipynb](analysis/outcome_correlation.ipynb)


## Citation

If you use this code or data, please cite our work:

```bibtex
@inproceedings{rab-cred_2026,
    title = "LLMs as annotators of credibility assessment in Danish asylum decisions: evaluating classification performance and errors beyond aggregated metrics",
    author = "Galadrielle Humblot-Renaux and Mohammad Naser Sabet Jahromi and Rohat Bakuri-Jørgensen and Marieke Anne Heyl and Asta S. Stage Jarlner and Maria Vlachou and Anna Murphy Høgenhaug and Desmond Elliott and Thomas Gammeltoft-Hansen and Thomas B. Moeslund",
    booktitle = "Proceedings of the 20th Linguistic Annotation Workshop (LAW-XX)",
    year = "2026",
    publisher = "Association for Computational Linguistics"
}
```

## Contact

For questions or issues, please contact Galadrielle Humblot-Renaux - https://vbn.aau.dk/en/persons/gegeh/ or create an issue in this repo.

## Acknowledgments

This work was supported by the Villum Foundation (“XAICRED”, grant no. 69198), the Grundfos Foundation (“REPAI”, grant no. 83648813), and the Danish National Research Foundation ("Center of Excellence for Global Mobility Law", grant no. DNRF169).

Part of the computation done for this project was performed on the UCloud interactive HPC system, which is managed by the eScience Center at the University of Southern Denmark. Part of the computation was also performed on the AI Cloud HPC system managed by CLAAUDIA at Aalborg University. 
