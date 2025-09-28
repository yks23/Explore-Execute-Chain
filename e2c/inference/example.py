from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "KaisenYang/Explore-Execute-Chain"
model_type = "8B-Final"  # change to the subfolder you want to use

tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder=model_type)
model = AutoModelForCausalLM.from_pretrained(model_name, subfolder=model_type)

# Test example: Fibonacci sequence
inputs = tokenizer("What is the 10th number in the Fibonacci sequence?", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
