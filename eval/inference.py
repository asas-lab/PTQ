from transformers import AutoModelForCausalLM, AutoTokenizer


# load the quntiazed model from asas HF

def generate_from_model(model_id, tokenizer, prompt, max_new_tokens):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto"),
    encoded_input = tokenizer(prompt, return_tensors='pt')
    output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda(), max_new_tokens=max_new_tokens)
    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)