import logging
from typing import get_args
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from omegaconf import OmegaConf

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

import torch

from .schemas import SCHEMA_MAP

THINKING_MODELS = ["Qwen/Qwen3-32B","deepcogito/cogito-v1-preview-qwen-14B","Qwen/Qwen3-8B"]

def load_prompt(CONFIG_LLM):
    # load prompt from external txt file
    with open(CONFIG_LLM.prompt.template_path, "r", encoding="utf-8") as f:
        credibility_prompt = f.read()
    assert CONFIG_LLM.prompt.placeholder in credibility_prompt, f"Prompt must contain {CONFIG_LLM.prompt.placeholder} placeholder"
    return credibility_prompt
    
def load_system_prompt(CONFIG_LLM):
    # load prompt from external txt file
    with open(CONFIG_LLM.system_prompt.template_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    return system_prompt
    
def load_llm_model(CONFIG_LLM):
    global model, tokenizer
    schema = SCHEMA_MAP[CONFIG_LLM.output.schema]
    if CONFIG_LLM.library == "hf":
        if "mistralai/Mistral-Small" in CONFIG_LLM.model.name:
            tokenizer = MistralTokenizer.from_hf_hub(CONFIG_LLM.model.name)
            model = Mistral3ForConditionalGeneration.from_pretrained(
                CONFIG_LLM.model.name, device_map=CONFIG_LLM.model.device
            )
        elif "mistralai/Ministral" in CONFIG_LLM.model.name:
            model = Mistral3ForConditionalGeneration.from_pretrained(CONFIG_LLM.model.name, device_map=CONFIG_LLM.model.device)
            tokenizer = MistralCommonBackend.from_pretrained(CONFIG_LLM.model.name)

        elif "bigscience" in CONFIG_LLM.model.name:
            chat_template="""{%- for message in messages %}{{- message['content'] + '\n\n'}}{%- endfor %}"""
            if "mt0" in CONFIG_LLM.model.name:
                model = outlines.from_transformers(
                    AutoModelForSeq2SeqLM.from_pretrained(CONFIG_LLM.model.name, device_map=CONFIG_LLM.model.device),
                    AutoTokenizer.from_pretrained(CONFIG_LLM.model.name, chat_template=chat_template)
                )
            else:
                model = outlines.from_transformers(
                        AutoModelForCausalLM.from_pretrained(CONFIG_LLM.model.name, device_map=CONFIG_LLM.model.device),
                        AutoTokenizer.from_pretrained(CONFIG_LLM.model.name, chat_template=chat_template)
                    )
        elif "aya-101" in CONFIG_LLM.model.name:
            model = outlines.from_transformers(
                    AutoModelForSeq2SeqLM.from_pretrained(CONFIG_LLM.model.name, device_map=CONFIG_LLM.model.device),
                    AutoTokenizer.from_pretrained(CONFIG_LLM.model.name)
                )
        elif "Qwen3-VL" in CONFIG_LLM.model.name:
            model = outlines.from_transformers(
                    Qwen3VLForConditionalGeneration.from_pretrained(CONFIG_LLM.model.name, device_map=CONFIG_LLM.model.device),
                    AutoProcessor.from_pretrained(CONFIG_LLM.model.name)
                )
        else:
            model = outlines.from_transformers(
                    AutoModelForCausalLM.from_pretrained(CONFIG_LLM.model.name, device_map=CONFIG_LLM.model.device),
                    AutoTokenizer.from_pretrained(CONFIG_LLM.model.name)
                )
    return model

def parse_prompt_params(prompt):
    # split prompt template into multiple parts
    prompt_template_parts = prompt.split("=================")
    prompts_params = []
    prompts = []
    # extract prompts and prompt parameters
    for i in range(1,len(prompt_template_parts)):
        if i%2 != 0: 
            prompts_params.append(prompt_template_parts[i])
        else:
            prompts.append(prompt_template_parts[i])
            
    assert len(prompts_params) == len(prompts), "Number of prompt params and prompts must be equal."
    
    return prompts_params, prompts

def get_llm_output(prompt_template, case_text, CONFIG_LLM, system_prompt=None):
    raw_prompt = prompt_template.replace(CONFIG_LLM.prompt.placeholder, case_text)
    
    final_schema = SCHEMA_MAP[CONFIG_LLM.output.schema]
    final_assessment_dict = dict()
    
    prompts_params, prompts = parse_prompt_params(raw_prompt)
    assert len(prompts) == 1, f"Prompt is multi-step or is missing a params section"
    
    prompt = prompts[0]
    prompt_params = prompts_params[0]
    prompt_params_yaml = OmegaConf.create(prompt_params)
    
    output_schema = SCHEMA_MAP[prompt_params_yaml.output_schema]
    credibility_schema_field = prompt_params_yaml.get("credibility_schema_field", None)
    
    constrain_output = not (output_schema == str or "mistralai" in CONFIG_LLM.model.name or CONFIG_LLM.model.name in ["mistralai/Mistral-7B-Instruct-v0.3","speakleash/Bielik-11B-v3.0-Instruct","utter-project/EuroLLM-22B-Instruct-2512","speakleash/Bielik-11B-v2.3-Instruct",
                                                                              "utter-project/EuroLLM-9B-Instruct-2512"])
    
    messages = [
        ]
    if system_prompt is not None and len(system_prompt):
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    messages.append({
            "role": "user",
            "content": prompt
        })
        
        
    if CONFIG_LLM.library == "hf":
        model_config_dict = OmegaConf.to_container(CONFIG_LLM.model)
        model_config_dict.pop("device")
        model_config_dict.pop("name")
        #try:
        #generator = outlines.Generator(model, output_schema)
        if constrain_output:
            output = model(
                outlines.inputs.Chat(messages),
                output_type=output_schema,
                **model_config_dict
            )
        elif "Ministral" in CONFIG_LLM.model.name:
            tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)
            tokenized["input_ids"] = tokenized["input_ids"].to(device=model.device)
            tokenized["attention_mask"] = tokenized["attention_mask"].to(device=model.device)
            
            raw_output = model.generate(
                **tokenized,
                **model_config_dict
            )[0]
            output = tokenizer.decode(raw_output[len(tokenized["input_ids"][0]):])
            
        elif "Mistral-Small" in CONFIG_LLM.model.name:
            tokenized = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))
            input_ids = torch.tensor([tokenized.tokens]).to(model.device)
            attention_mask = torch.ones_like(input_ids).to(model.device)
            raw_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **model_config_dict
            )[0]
            output = tokenizer.decode(raw_output[len(tokenized.tokens) :])
        else:
            #print(f"!!! LLM output schema is not constrained, getting raw string output.")
            output = model(
                outlines.inputs.Chat(messages),
                **model_config_dict
            )
        response = output.strip()

        if output_schema != str:
            if set(get_args(output_schema)) == {"POSITIVE CREDIBILITY ASSESSMENT", "NEGATIVE CREDIBILITY ASSESSMENT", "NO CREDIBILITY ASSESSMENT"}:
                assert "POSITIVE" in response or "NEGATIVE" in response or "NO " in response, f"LLM output '{response}' could not be mapped to 3-class prediction."
                response = "POSITIVE CREDIBILITY ASSESSMENT" if "POSITIVE" in response else "NEGATIVE CREDIBILITY ASSESSMENT" if "NEGATIVE" in response else "NO CREDIBILITY ASSESSMENT"
            assert response in get_args(output_schema), f"LLM output could not be validated using the specified schema {response}"
            # assessment = schema.model_validate_json(output)    
        
        final_assessment_dict[credibility_schema_field] = response
    else: raise NotImplementedError(f"LLM output parsing not implemented for library {CONFIG_LLM.library} yet.")
    
    final_assessment = final_schema.model_validate(final_assessment_dict)
    
    return final_assessment

def get_llm_output_multistep(prompt_template, case_text, CONFIG_LLM, system_prompt=None):
    prompt = prompt_template.replace(CONFIG_LLM.prompt.placeholder, case_text)
    final_schema = SCHEMA_MAP[CONFIG_LLM.output.schema]
    final_assessment_dict = dict()
    
    model_config_dict = OmegaConf.to_container(CONFIG_LLM.model)
    model_config_dict.pop("device")
    model_config_dict.pop("name")
    
    prompts_params, prompts = parse_prompt_params(prompt)
            
    messages = []
    if system_prompt is not None and len(system_prompt):
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
    responses = []
                
    if CONFIG_LLM.library == "ollama":
        
        raise NotImplementedError("Multistep LLM output not implemented for Ollama library yet.")
                
        
    elif CONFIG_LLM.library == "hf":
        #raise NotImplementedError("Multistep LLM output not implemented for Huggingface library yet.") 

        for prompt_params, prompt, prompt_idx in zip(prompts_params, prompts, range(len(prompts))):
            skip_prompt = False
            
            # parse params as YAML
            prompt_params_yaml = OmegaConf.create(prompt_params)
            output_schema = SCHEMA_MAP[prompt_params_yaml.output_schema]
            
            constrain_output = not (output_schema == str or "mistralai" in CONFIG_LLM.model.name or CONFIG_LLM.model.name in ["mistralai/Mistral-7B-Instruct-v0.3","speakleash/Bielik-11B-v3.0-Instruct","utter-project/EuroLLM-22B-Instruct-2512",
                                                                                      "utter-project/EuroLLM-9B-Instruct-2512"])
        
            conditions = prompt_params_yaml.get("conditions", None)
            credibility_schema_field = prompt_params_yaml.get("credibility_schema_field", None)
            #print(prompt_params_yaml, conditions)
            
            # check whether to skip this prompt based on conditions
            if conditions is not None:
                assert len(responses) > 0, "Conditions can only be applied from the second step onwards."
                # get LLM response from previous step
                response_prev = responses[-1]
                #print(response_prev)
                # extract condition attributes from prompt_params
                for cond_attr, cond_value in conditions.items():
                    # get value from previous response
                    value_prev = response_prev.get(cond_attr, None)
                    if value_prev != cond_value:
                        skip_prompt = True
            
            if not skip_prompt:
                messages.append({'role': 'user', 'content': prompts[prompt_idx]})

                #try:
                #generator = outlines.Generator(model, output_schema)
                if constrain_output:
                    output = model(
                        outlines.inputs.Chat(messages),
                        output_type=output_schema,
                        **model_config_dict
                    )
                elif "Ministral" in CONFIG_LLM.model.name:
                    tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)
                    tokenized["input_ids"] = tokenized["input_ids"].to(device=model.device)
                    tokenized["attention_mask"] = tokenized["attention_mask"].to(device=model.device)
                    
                    raw_output = model.generate(
                        **tokenized,
                        **model_config_dict
                    )[0]
                    output = tokenizer.decode(raw_output[len(tokenized["input_ids"][0]):])
                elif "Mistral-Small" in CONFIG_LLM.model.name:
                    tokenized = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))
                    input_ids = torch.tensor([tokenized.tokens]).to(model.device)
                    attention_mask = torch.ones_like(input_ids).to(model.device)
                    raw_output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **model_config_dict
                    )[0]
                    output = tokenizer.decode(raw_output[len(tokenized.tokens) :])
                else:
                    #print(f"!!! LLM output schema is not constrained, getting raw string output.")
                    output = model(
                        outlines.inputs.Chat(messages),
                        **model_config_dict
                    )
                    output = output.strip()
                    
                if output_schema != str:
                    if set(get_args(output_schema)) == {"Y", "N"}:
                        output = output[0]
                    elif set(get_args(output_schema)) == {"POSITIVE", "NEGATIVE"}:
                        if output[0] in ["P", "N"]:
                            output = "POSITIVE" if output[0] == "P" else "NEGATIVE"
                        else:
                            if "POSITIVE" in output and "NEGATIVE" in output:
                                if output.count("POSITIVE") > output.count("NEGATIVE"): output="POSITIVE"
                                else: output="NEGATIVE"
                            else:
                                if "POSITIVE" in output: output="POSITIVE"
                                else: output="NEGATIVE"
                    elif set(get_args(output_schema)) == {"POSITIVE CREDIBILITY ASSESSMENT", "NEGATIVE CREDIBILITY ASSESSMENT", "NO CREDIBILITY ASSESSMENT"}:
                        if not ("POSITIVE" in output or "NEGATIVE" in output or "NO " in output):
                            logging.warning(f"LLM output '{output}' could not be mapped to 3-class prediction.")
                            output = "NEGATIVE CREDIBILITY ASSESSMENT"
                        else:
                            output = "POSITIVE CREDIBILITY ASSESSMENT" if "POSITIVE" in output else "NEGATIVE CREDIBILITY ASSESSMENT" if "NEGATIVE" in output else "NO CREDIBILITY ASSESSMENT"
                    assert output in get_args(output_schema), f"LLM output could not be validated using the specified schema '{output}'"
                else:
                    logging.info(output)
                    
                messages.append({'role': 'assistant', 'content': output})
                
                response_dict = {
                    credibility_schema_field: output
                }
                responses.append(response_dict)
                
                #print(responses)
            
                if credibility_schema_field is not None:
                    final_assessment_dict[credibility_schema_field] = output
    
    #print(final_assessment_dict)
    final_assessment = final_schema.model_validate(final_assessment_dict)
    return final_assessment
