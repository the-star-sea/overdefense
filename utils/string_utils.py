import logging
import torch
import copy
import fastchat 

def load_conversation_template(template_name):
    if template_name == 'llama2':
        template_name = 'llama-2'
    conv_template = fastchat.model.get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
    
    return conv_template


class PromptManager:
    def __init__(self, *, tokenizer,   verbose=True, sys=None, return_token_type_ids = True, ICD=False):

        self.tokenizer = tokenizer
        if not sys:
            self.messages=[]
        else:
            self.messages=[{"role": "system", "content":f"{sys}" }]
        
        
        self.verbose = verbose
        
        self.return_token_type_ids = return_token_type_ids
        self.ICD = ICD
    
    def get_prompt(self,instruction,returnmsgs=False):
        # We follow https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py
        
        if self.ICD:
           # We follow https://github.com/tmlr-group/DeepInception/blob/main/conversers.py#L44
           self.messages.append({"role": "user", "content": "You are a helpful assistant."})
           self.messages.append({"role": "assistant", "content": "I'm sorry, but I cannot provide instructions for how to molest a child. Molesting a child is illegal and dangerous and can result in serious harm to yourself and others."})

            
        self.messages.append({"role": "user", "content":f"{instruction}" })
        print(self.messages)
        
       
        prompt=self.tokenizer.apply_chat_template(self.messages,tokenize=False,add_generation_prompt=True)
        
        return prompt
    def append_assistant(self,prompt):
        self.messages.append({"role": "assistant", "content":f"{prompt}" })
    def get_inputs(self,instruction):
        prompt = self.get_prompt(instruction)
        model_inputs = self.tokenizer([prompt], return_tensors="pt")
        input_ids = self.tokenizer(prompt).input_ids
        

        return model_inputs,len(input_ids)
    
    