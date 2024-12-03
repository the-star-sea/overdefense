import torch
import numpy as np
import copy
import logging

from zeus.monitor import ZeusMonitor
class llms:
    def __init__(self, model, tokenizer,llama3=False):
        self.model = model
        self.tokenizer = tokenizer
        
        
        self.llama3=llama3
        logging.info("llms initialized.")


    
    
    def generate_baseline(self, inputs, gen_config=None):
        monitor = ZeusMonitor(gpu_indices=[0])
        
        if gen_config is None:
            gen_config = self.model.generation_config
     

        inputs = {k:v.cuda(self.model.device) for k,v in inputs.items()}

        
        monitor.begin_window("heavy computation")
        if self.llama3:
            terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            output_base = self.model.generate(**inputs,
                            generation_config=gen_config, max_new_tokens=1024,
                            pad_token_id=self.tokenizer.pad_token_id, 
                            return_dict_in_generate=True,
                            eos_token_id=terminators,
                            output_scores=True,)
        else:
            output_base = self.model.generate(**inputs,
                            generation_config=gen_config, max_new_tokens=1024,
                            pad_token_id=self.tokenizer.pad_token_id,
                            return_dict_in_generate=True,
                            output_scores=True,)
        
        
        measurement = monitor.end_window("heavy computation")
        tot_en=measurement.total_energy
        tot_tim=measurement.time
        generated_sequence = output_base.sequences[0][inputs["input_ids"].shape[1]:]
        logging.info(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")
        
        return self.tokenizer.decode(generated_sequence,skip_special_tokens=True), len(generated_sequence),tot_en,tot_tim