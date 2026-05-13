# add root folder to path
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import argparse
from datetime import datetime
from omegaconf import OmegaConf
import pathlib
import logging
import gc
import torch

from llm_annotation.schemas import CLASS_NAMES_3CLS
from llm_annotation.llm_utils import *
from llm_annotation.schemas import SCHEMA_MAP

# get command line arguments
config_cli = OmegaConf.from_cli()
print("Command-line arguments:")
print(OmegaConf.to_yaml(config_cli))

# load config file
config_cli.config_path = config_cli.get('config_path', 'experiment_configs/ACLW_val.yml')
CONFIG = OmegaConf.load(pathlib.Path(config_cli.config_path))

# override config with any command-line arguments
CONFIG = OmegaConf.merge(CONFIG, config_cli)
OmegaConf.set_readonly(CONFIG, True)

print("Loaded experiment configuration:")
print(OmegaConf.to_yaml(CONFIG))

# prepare results folder
config_name = pathlib.Path(config_cli.config_path).stem


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


checkpoint_every = int(
    config_cli.get("checkpoint_every", OmegaConf.select(CONFIG, "results.checkpoint_every", default=100))
)
resume = _to_bool(config_cli.get("resume", OmegaConf.select(CONFIG, "results.resume", default=False)))
resume_results_folder = config_cli.get(
    "resume_results_folder", OmegaConf.select(CONFIG, "results.resume_results_folder", default=None)
)


# Keys that identify a run's configuration vs. runtime/meta flags
_TRANSIENT_KEYS = {"resume", "resume_results_folder", "checkpoint_every", "config_path"}


def _experiment_config_dict(cfg):
    """Return a plain dict with transient/runtime keys stripped for comparison."""
    d = OmegaConf.to_container(cfg, resolve=True)
    for key in _TRANSIENT_KEYS:
        d.pop(key, None)
    if "results" in d and isinstance(d["results"], dict):
        for key in _TRANSIENT_KEYS:
            d["results"].pop(key, None)
    return d


def _config_matches(folder, current_cfg):
    """Return True if folder/config.yml has the same experiment config as current_cfg."""
    saved_config_path = pathlib.Path(folder) / "config.yml"
    if not saved_config_path.exists():
        return False
    try:
        saved_cfg = OmegaConf.load(saved_config_path)
        return _experiment_config_dict(saved_cfg) == _experiment_config_dict(current_cfg)
    except Exception:
        return False


def _find_latest_resume_folder(results_root, run_name_prefix, current_cfg):
    results_root = pathlib.Path(results_root)
    candidate_folders = [
        folder for folder in results_root.glob(f"{run_name_prefix}_*")
        if folder.is_dir() and (folder / "preds.csv").exists()
        and _config_matches(folder, current_cfg)
    ]
    if not candidate_folders:
        return None
    return max(candidate_folders, key=lambda folder: folder.stat().st_mtime)

if resume:
    if resume_results_folder:
        candidate = pathlib.Path(resume_results_folder)
        if not _config_matches(candidate, CONFIG):
            print(
                f"WARNING: Config in {candidate} does not match current config. "
                "Starting a new run instead of resuming."
            )
            resume = False
        else:
            results_folder = candidate
    if resume:  # may have been reset above
        if not resume_results_folder:  # auto-detect branch
            latest_folder = _find_latest_resume_folder(CONFIG.results.folder, config_name, CONFIG)
            if latest_folder is None:
                print(
                    "Resume requested but no matching prior run found "
                    "(no folder with a compatible config.yml). Starting a new run."
                )
                resume = False
            else:
                results_folder = latest_folder
                print(f"Resume requested. Using matching results folder: {results_folder}")

if not resume:
    timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    results_folder = pathlib.Path(f"{CONFIG.results.folder}/{config_name}_{timestamp:}")

results_folder.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,     # Log level (INFO and above)
    handlers=[
        logging.FileHandler(results_folder / "output.log"),
        logging.StreamHandler()
    ]
)

# load human-labeled cases
df_human = pd.read_csv(CONFIG.dataset.human_csv_path, encoding="utf-8-sig").dropna(axis=0, how='all') # read CSV & drop empty rows
# --- FIX COLUMN NAMES ---
df_human.columns = df_human.columns.str.strip()

df_human = df_human.fillna("")

# load prompt template, model, and dataset schema
system_prompt = load_system_prompt(CONFIG.llm)
prompt_template = load_prompt(CONFIG.llm)

# check that the prompt template contains every class in CLASS_NAMES_3CLS
# for class_name in CLASS_NAMES_3CLS.values():
#     if class_name not in prompt_template:
#         logging.error(f"!! Class name '{class_name}' not found in prompt template. Please check that the prompt template contains all class names.")
#         raise ValueError(f"Class name '{class_name}' not found in prompt template.")

load_llm_model(CONFIG.llm)
schema_human = SCHEMA_MAP[CONFIG.dataset.human_schema]


def _normalize_case_idx(case_idx):
    if pd.isna(case_idx):
        return ""
    if isinstance(case_idx, float) and case_idx.is_integer():
        return str(int(case_idx))
    value = str(case_idx).strip()
    if value.endswith(".0"):
        numeric_part = value[:-2]
        if numeric_part.lstrip("-").isdigit():
            return numeric_part
    return value


# iterate over the human-labeled cases, get the LLM output and compare the human vs. LLM predictions
preds_path = results_folder / "preds.csv"
preds_buffer = {
    "case_idxs": [],
    "human_preds": [],
    "human_confs": [],
    "llm_preds": [],
    #"llm_confs": [],
    "llm_reasoning": []
}


def _append_buffer_to_csv(buffer_dict, save_path):
    if len(buffer_dict["case_idxs"]) == 0:
        return
    df = pd.DataFrame.from_dict(buffer_dict)
    write_header = not save_path.exists()
    df.to_csv(save_path, mode="a", header=write_header, index=False)


def _clear_buffer(buffer_dict):
    for key in buffer_dict:
        buffer_dict[key].clear()


existing_processed_count = 0
processed_case_idxs = set()
if resume and preds_path.exists():
    existing_case_idxs_df = pd.read_csv(preds_path, encoding="utf-8-sig", usecols=["case_idxs"]).dropna(axis=0, how='all')
    existing_case_idxs_df.columns = existing_case_idxs_df.columns.str.strip()
    existing_case_idxs = existing_case_idxs_df["case_idxs"].tolist()
    processed_case_idxs = {_normalize_case_idx(case_idx) for case_idx in existing_case_idxs}
    existing_processed_count = len(processed_case_idxs)
    logging.info(f"Loaded {existing_processed_count} existing case indices from {preds_path}.")


def persist_checkpoint():
    _append_buffer_to_csv(preds_buffer, preds_path)
    _clear_buffer(preds_buffer)
    OmegaConf.save(CONFIG, results_folder / "config.yml")

unsaved_since_checkpoint = 0

system_prompt_variant = CONFIG.llm.system_prompt.template_path.split("/")[-1].replace(".txt","")
user_prompt_variant = CONFIG.llm.prompt.template_path.split("/")[-1].replace(".txt","")
    
try:
    for i, row in df_human.iterrows():

        try:
            human_assessment = schema_human.model_validate(row.to_dict())
            
            logging.info(f"Case {human_assessment.idx} ({i+1}/{len(df_human)}) [LLM {CONFIG.llm.model.name}, System Prompt: {system_prompt_variant}, User Prompt: {user_prompt_variant}]:")
        except ValueError as e:
            logging.warning(f"!! Case with Index {row['Index']} could not be validated using {schema_human.__name__} schema: {e}, skipping")
            continue

        case_idx_norm = _normalize_case_idx(human_assessment.idx)
        if case_idx_norm in processed_case_idxs:
            logging.info(f"Case {human_assessment.idx} already processed. Skipping.")
            continue
        
        human_assessment_3cls = human_assessment.map_to_3cls_prediction()
        human_assessment_conf = human_assessment.map_to_confidence()
        
        is_multistep = len(prompt_template.split("=================")) > 3 # check how many headers in prompt
        
        # start a timer
        start_time = datetime.now()
        try:
            if not is_multistep:
                llm_assessment = get_llm_output(prompt_template, human_assessment.case_text, CONFIG.llm, system_prompt=system_prompt)
            else:
                llm_assessment = get_llm_output_multistep(prompt_template, human_assessment.case_text, CONFIG.llm, system_prompt=system_prompt)
                #raise NotImplementedError
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.error(f"OOM on case {human_assessment.idx}. Skipping this case and continuing.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue
            raise
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        logging.info(f"-> LLM response time: {elapsed_time:.2f} seconds")

    #if not CONFIG.llm.prompt.get('is_multistep',False):
        #    llm_assessment = get_llm_output(prompt_template, human_assessment.case_text, CONFIG.llm)
        #else:
        #    llm_assessment = get_llm_output_multistep(prompt_template, human_assessment.case_text, CONFIG.llm)
        llm_pred_3cls = llm_assessment.map_to_3cls_prediction()
        #llm_pred_conf = llm_assessment.map_to_confidence()
        #llm_pred_justification = llm_assessment.map_to_justification()
        
        logging.info(f"-> Human: {human_assessment_3cls}")
        logging.info(f"-> LLM:: {llm_pred_3cls}")
    # logging.info(f"-> LLM Justification: {llm_pred_justification}")
        
        preds_buffer["human_confs"].append(human_assessment_conf)
        preds_buffer["human_preds"].append(human_assessment_3cls)
        
        preds_buffer["llm_preds"].append(llm_pred_3cls)
        # check if llm_assessment has attribute 'confidence' (it may not have, depending on schema)
        #preds["llm_confs"].append(llm_pred_conf)

        preds_buffer["llm_reasoning"].append(llm_assessment.reasoning)
        
        preds_buffer["case_idxs"].append(human_assessment.idx)
        processed_case_idxs.add(case_idx_norm)
        unsaved_since_checkpoint += 1

        if unsaved_since_checkpoint >= checkpoint_every:
            persist_checkpoint()
            logging.info(
                f"Checkpoint saved to {preds_path} and {results_folder / 'config.yml'} "
                f"(processed {len(processed_case_idxs)} total cases)."
            )
            unsaved_since_checkpoint = 0
        
    persist_checkpoint()
    logging.info(f"Final checkpoint saved to {preds_path} and {results_folder / 'config.yml'}")

    # if the schema is different from ACLWCredibilityUnlabeled, also save a confusion matrix and the config file used for the experiment
    preds_df = pd.read_csv(preds_path, encoding="utf-8-sig").dropna(axis=0, how='all') if preds_path.exists() else pd.DataFrame()
    if CONFIG.dataset.human_schema != "ACLWCredibilityUnlabeled" and len(preds_df) > 0:
        # check if gt is a tuple or string
        human_preds = preds_df["human_preds"].tolist()
        llm_preds = preds_df["llm_preds"].tolist()
        if human_preds[0] is not None and isinstance(human_preds[0], tuple):
            gt = [pred[0] for pred in human_preds]
        else:
            gt = human_preds

except Exception as e:
    if len(preds_buffer["case_idxs"]) > 0:
        persist_checkpoint()
        logging.info(
            f"Saved partial checkpoint to {preds_path} and {results_folder / 'config.yml'} after exception."
        )
    logging.error(f"An error occurred during processing: {e}, stopping execution.")
