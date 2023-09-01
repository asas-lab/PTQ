import transformers, torch, datasets, numpy
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import List 
import argparse, os



def compute_prelpexity(model_id: str, testset: str) -> tuple:

  """
    Input:
        model_id: HF model id
        testset: a path to testset in CSV format
    Return:
        a tuple with all preplexity socres in a list and their averge 
    """

  model = AutoModelForCausalLM.from_pretrained(model_id)
  tokenizer = AutoTokenizer.from_pretrained(model_id)

  ppls = []
  
  for text in testset: 
    input = tokenizer(text, return_tensors='pt')
    loss = model(input_ids = input['input_ids'], 
              labels = input['input_ids']).loss
    ppl = torch.exp(loss)
    ppls.append(ppl.item())

  avg = numpy.average(ppls)
  avg = round(avg, 2)
  return (ppls, avg)


parser = argparse.ArgumentParser(description = "compute preplexity")
parser.add_argument("model", type=str, help="huggingface model id")
parser.add_argument("testset", type=str, help="path to testset in CSV file fromat with the data in the first col")
parser.add_argument("--hf_dataset", type=str, help="hugging face dataset id")

args = parser.parse_args()


def main():

    testset_path = os.path.join(args.testset)
    df = pd.read_csv(testset_path)
    compute_prelpexity(args.model, df[df.columns[0]])

    ##TODOS
    if args.hf_dataset:
       pass
       

if __name__ == '__main__':
    main()