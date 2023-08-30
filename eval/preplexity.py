import transformers, torch, datasets, numpy
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import List 



def compute_prelpexity(model_id: str, dataset: List) -> tuple:

  model = AutoModelForCausalLM.from_pretrained(model_id)
  tokenizer = AutoTokenizer.from_pretrained(model_id)

  ppls = []
  
  for text in dataset: 
    input = tokenizer(text, return_tensors='pt')
    loss = model(input_ids = input['input_ids'], 
              labels = input['input_ids']).loss
    ppl = torch.exp(loss)
    ppls.append(ppl.item())

  avg = numpy.average(ppls)
  avg = round(avg, 2)
  return (ppls, avg)
