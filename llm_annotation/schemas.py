# add root folder to path
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from pydantic import BaseModel, Field
from typing import Literal, Optional

SCHEMA_MAP = {}
def register(cls):
  SCHEMA_MAP[cls.__name__] = cls
  return cls

CLASS_NAMES_3CLS = {
    "absent": "NO CREDIBILITY ASSESSMENT",
    "pos": "POSITIVE CREDIBILITY ASSESSMENT",
    "neg": "NEGATIVE CREDIBILITY ASSESSMENT"
}

SCHEMA_MAP["3ClassCredibility"] = Literal[
    CLASS_NAMES_3CLS["absent"],
    CLASS_NAMES_3CLS["pos"],
    CLASS_NAMES_3CLS["neg"]
]
SCHEMA_MAP["YesNo"] = Literal["Y","N"]
SCHEMA_MAP["PosNeg"] = Literal["POSITIVE","NEGATIVE"]
SCHEMA_MAP["FreeText"] = str

@register
class Text(BaseModel):
    '''
    Basic text output schema for the LLM output.
    '''
    text: str
      
    
@register
class ACLWCredibilityHuman(BaseModel):
    '''
    Schema for human assessments (extracted from the CSV) with single motive only.
    Cases with multiple motives will raise a ValueError during validation and thus be skipped.
    The field names match the CSV column names.
    '''
    case_text: str = Field(alias='Text/Decision')
    idx: float = Field(alias='Index')
    # IF MULTPLE MOTIVES SHOULD BE SKIPPED AGAIN: activate the line below
    #multi_motives: Literal["N"] = Field(alias='Several motives')
    q1: Literal["Y","N"] = Field(alias='Question 1: Credibility factoring into reasoning')
    q2: Literal["POSITIVE", "NEGATIVE","-"] = Field(alias='Question 2: Credibility positive or negative?')
    confidence_q1: Literal["LOW","MEDIUM","MIDDLE","HIGH"] = Field(alias='Confidence Q1')
    confidence_q2 : Literal["LOW","MEDIUM","MIDDLE","HIGH","-"] = Field(alias='Confidence Q2')
    
    
    class Config:
        validate_by_name = True
    
    def map_to_3cls_prediction(self):
        human_chosen_option = CLASS_NAMES_3CLS["absent"] if self.q1 == "N" else CLASS_NAMES_3CLS["pos"] if self.q2 == "POSITIVE" else CLASS_NAMES_3CLS["neg"]
        return human_chosen_option
        
    def map_to_confidence(self):
        return (self.confidence_q1, self.confidence_q2)  
    
    
@register
class ACLWCredibilityHumanMultiAnn(BaseModel):
    '''
    Schema for human assessments (extracted from the CSV) with single motive only.
    Cases with multiple motives will raise a ValueError during validation and thus be skipped.
    The field names match the CSV column names.
    '''
    case_text: str = Field(alias='Text/Decision')
    idx: float = Field(alias='Index')
    # IF MULTPLE MOTIVES SHOULD BE SKIPPED AGAIN: activate the line below
    #multi_motives: Literal["N"] = Field(alias='Several motives')
    q1_H1: Literal["Y","N"] = Field(alias='Question 1: Credibility factoring into reasoning_H1')
    q1_H2: Literal["Y","N"] = Field(alias='Question 1: Credibility factoring into reasoning_H2')
    q1_H3: Literal["Y","N", ""] = Field(alias='Question 1: Credibility factoring into reasoning_H3', default=None) # resolver, optional because some cases were only annotated by 2 annotators
    q2_H1: Literal["POSITIVE", "NEGATIVE","-"] = Field(alias='Question 2: Credibility positive or negative?_H1')
    q2_H2: Literal["POSITIVE", "NEGATIVE","-"] = Field(alias='Question 2: Credibility positive or negative?_H2')
    q2_H3: Literal["POSITIVE", "NEGATIVE","-", ""] = Field(alias='Question 2: Credibility positive or negative?_H3', default=None), # resolver, optional because some cases were only annotated by 2 annotators
    
    confidence_q1_H1: Literal["LOW","MEDIUM","MIDDLE","HIGH"] = Field(alias='Confidence q1_H1')
    confidence_q1_H2: Literal["LOW","MEDIUM","MIDDLE","HIGH"] = Field(alias='Confidence q1_H2')
    confidence_q1_H3: Literal["LOW","MEDIUM","MIDDLE","HIGH", ""] = Field(alias='Confidence Q1_H3', default=None) # resolver, optional because some cases were only annotated by 2 annotators
    confidence_q2_H1 : Literal["LOW","MEDIUM","MIDDLE","HIGH","-"] = Field(alias='Confidence Q2_H1')
    confidence_q2_H2 : Literal["LOW","MEDIUM","MIDDLE","HIGH","-"] = Field(alias='Confidence Q2_H2')
    confidence_q2_H3 : Literal["LOW","MEDIUM","MIDDLE","HIGH","-", ""] = Field(alias='Confidence Q2_H3', default=None) # resolver, optional because some cases were only annotated by 2 annotators
    
    
    class Config:
        validate_by_name = True
    
    def map_to_3cls_prediction(self):
        human_chosen_option_H1 = CLASS_NAMES_3CLS["absent"] if self.q1_H1 == "N" else CLASS_NAMES_3CLS["pos"] if self.q2_H1 == "POSITIVE" else CLASS_NAMES_3CLS["neg"]
        human_chosen_option_H2 = CLASS_NAMES_3CLS["absent"] if self.q1_H2 == "N" else CLASS_NAMES_3CLS["pos"] if self.q2_H2 == "POSITIVE" else CLASS_NAMES_3CLS["neg"]
        if self.q1_H3 and self.q2_H3:
            human_chosen_option_H3 = CLASS_NAMES_3CLS["absent"] if self.q1_H3 == "N" else CLASS_NAMES_3CLS["pos"] if self.q2_H3 == "POSITIVE" else CLASS_NAMES_3CLS["neg"]
        else:
            human_chosen_option_H3 = None
        return (human_chosen_option_H1, human_chosen_option_H2, human_chosen_option_H3)
        
    def map_to_confidence(self):
        return ((self.confidence_q1_H1, self.confidence_q2_H1), (self.confidence_q1_H2, self.confidence_q2_H2), (self.confidence_q1_H3, self.confidence_q2_H3))  

   

@register
class ACLWCredibility(BaseModel):
    '''
    Basic schema for the LLM output. Designed to be used with llm_prompts/DEMO.txt
    '''
    credibility_3cls: SCHEMA_MAP["3ClassCredibility"] = None
    credibility_present: SCHEMA_MAP["YesNo"] = None
    credibility_sentiment: SCHEMA_MAP["PosNeg"] = None
    reasoning: str = None
    
    def map_to_3cls_prediction(self):
        if self.credibility_3cls is not None:
            return self.credibility_3cls
        
        if self.credibility_present == "N":
            return CLASS_NAMES_3CLS["absent"]
        elif self.credibility_sentiment == "POSITIVE":
            return CLASS_NAMES_3CLS["pos"]
        elif self.credibility_sentiment == "NEGATIVE":
            return CLASS_NAMES_3CLS["neg"]
        else:
            print("!! Could not map LLM output to 3-class prediction.")
            print(self)
            raise ValueError