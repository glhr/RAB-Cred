import pathlib
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import re

from sklearn.metrics import f1_score, cohen_kappa_score

from paths import *

def get_aggregated_dataframe(experiment_folders, exclude_idxs=None, subset="test"):
    result_dicts = []

    result_dfs_raw = []
    # iterate over experiment folders and print the names
    for folder in experiment_folders:

        if not (pathlib.Path(folder) / "config.yml").exists() or not (pathlib.Path(folder) / "preds.csv").exists():
            continue
        
        # load the config file
        CONFIG = OmegaConf.load(pathlib.Path(folder) / "config.yml")
        OmegaConf.set_readonly(CONFIG, True)
        
        model_name = CONFIG.llm.model.name
        system_prompt = CONFIG.llm.system_prompt.template_path.split("/")[-1].replace(".txt","")
        user_prompt = CONFIG.llm.prompt.template_path.split("/")[-1].replace(".txt","")
        max_new_tokens = CONFIG.llm.model.get("max_new_tokens", None)
        config_path = CONFIG.get("config_path", "unknown")
        if isinstance(CONFIG.llm.model.get("temperature", "default"), float):
            temperature = "default"
        else:
            temperature = str(CONFIG.llm.model.get("temperature", "default"))
        do_sample = CONFIG.llm.model.get("do_sample", None)
        
        
        # extract number of parameters from model name with regex
        num_params = re.search(r"-(\d+)([Bb])-", model_name)
        if num_params is None:
            num_params = num_params_per_model.get(model_name, "unknown")
            if num_params == "unknown":
                raise ValueError(f"Number of parameters for model {model_name} not found in predefined dictionary.")
        else:
            num_params = float(num_params.group(0)[1:-2])
        
        # read the preds csv as dataframe
        df_preds = pd.read_csv(pathlib.Path(folder) / "preds.csv", encoding="utf-8")
        
        if subset == "val" and exclude_idxs is not None:
            df_preds = df_preds[~df_preds["case_idxs"].isin(exclude_idxs)]
        
        if subset == "test": # multiple human annotations
            #NOTE  human preds format: ('NEGATIVE CREDIBILITY ASSESSMENT', 'NEGATIVE CREDIBILITY ASSESSMENT', None)
            humans_withoutnone = [ [h for h in eval(p) if h is not None] for p in df_preds["human_preds"] ]
            
            human_maj = [max(h, key=h.count) for h in humans_withoutnone]
            human_1 = [h[0] for h in humans_withoutnone]
            assert (np.array(human_maj) == np.array(human_1)).sum() > 194, (np.array(human_maj) == np.array(human_1)).sum()
            
            humans_agree = [len(set(humans)) == 1 for humans in humans_withoutnone]
        elif subset == "val": # only one human annotation available
            humans_withoutnone = [ [p] for p in df_preds["human_preds"] ]
            human_maj = df_preds["human_preds"].tolist()
            humans_agree = [True] * len(human_maj) # dummy value, not used in val set analysis
        
        llm = df_preds["llm_preds"]
        
        is_correct_multi = [True if llm in h else False for llm, h in zip(llm, humans_withoutnone)]
        is_correct_maj = [llm == h for llm, h in zip(llm, human_maj)]
        
        
        model_config = {
                "experiment_folder": folder.split("/")[-1],
                "max_new_tokens": max_new_tokens,
                "System Prompt": system_prompt,
                "User Prompt": user_prompt,
                "model_name": model_name,
                "num_params": num_params,
                "config_path": config_path,
                "temperature": temperature
        }
        
        if (subset == "test" and not len(df_preds) == 200) or (subset == "val" and not len(df_preds) == 70):
            print(f"Skipping experiment with model {model_name} and prompt {user_prompt} due to incomplete predictions ({len(df_preds)} rows).")
            continue
        
        if subset == "test":
            performance_dict = {
                "f1_score (vs. human maj.)": f1_score(human_maj, llm, average='macro'),
                # kappa coefficient
                "kappa (vs. human maj.)": cohen_kappa_score(human_maj, llm),
                #"f1_score_cls": tuple(f1_score(human, llm, average=None)),
                "accuracy": np.array(human_maj == llm).mean()
            }
            df_preds["is_correct_multi"] = is_correct_multi
            df_preds["is_correct_maj"] = is_correct_maj
            df_preds["humans_agree"] = humans_agree
            df_preds["human_maj"] = human_maj
        elif subset == "val":
            performance_dict = {
                "f1_score": f1_score(human_maj, llm, average='macro'),
                # kappa coefficient
                "kappa": cohen_kappa_score(human_maj, llm),
                #"f1_score_cls": tuple(f1_score(human, llm, average=None)),
                "accuracy": np.array(human_maj == llm).mean()
            }
        
        result_dicts.append( {**model_config, "subset": "all", **performance_dict} )

        
        #df_preds = df_preds[["case_idxs", "human_preds", "human_confs", "llm_preds", "is_correct", "humans_agree"]]
        df_preds["idx_new"] = df_preds.index
        for key, value in model_config.items():
            df_preds[key] = value
        result_dfs_raw.append( df_preds )
        

    # create a dataframe from the result dicts
    df_results_aggregated = pd.DataFrame(result_dicts)
        
    assert len(df_results_aggregated) == len(result_dicts)

    if subset == "test":
        assert len(result_dicts) == 15, f"Expected 15 result dicts, but got {len(result_dicts)}. Check if all expected experiment folders were processed correctly."
        df_case = pd.concat(result_dfs_raw, ignore_index=True)

        assert len(df_case) == 200 * len(result_dicts) == 200 * 15

        # merge this with human_assessments/public_cleaned/test_set_merged_with_outcomes.csv
        test_set_gt = pd.read_csv(test_set_csv_metadata)
        test_set_gt["idx_new"] = test_set_gt.index

        # add the outcome column to df_case by merging on idx_new
        df_case_merged = pd.merge(left=df_case, right=test_set_gt, on="idx_new", how="left").drop(
            columns=["Index"] + [f"Q1: Credibility assessment presence (H{H})" for H in range(1,4)] + [f"Q2: Credibility assessment sentiment (H{H})" for H in range(1,4)] + [f"Confidence Q1 (H{H})" for H in range(1,3)] + [f"Confidence Q2 (H{H})" for H in range(1,3)]
        )
        assert len(df_case_merged) == len(df_case) == 200 * 15
        df_case.head()
        assert len(df_case) == 15*200, f"Expected {15*200} rows in df_case, but got {len(df_case)}. Check if all result dataframes were concatenated correctly."

        return df_results_aggregated, df_case, df_case_merged
    elif subset == "val":
        assert len(df_results_aggregated) == 630, f"Expected 630 results (21 models x 30 prompt combinations), but got {len(df_subset)}. Please check for duplicates or missing runs."
        return df_results_aggregated