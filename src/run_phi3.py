from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

torch.cuda.empty_cache()

# Replace this with the Hugging Face repository name
model_name = "microsoft/Phi-3.5-mini-instruct"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/")

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Example: Tokenize a prompt and generate a response
system_prompt = "System: You are a helpful assistant. Keep responses to at most a single sentence and concise. User: "
assistant_prompt = " Assistant:"
prompts = [
        "What is 17 times 23?",
        "What is the capital of France?",
        "What is the capital of Australia?",
        "What is the square root of 256?",
        "What is the population of New York City?",
        "What is the largest planet in the solar system?",
        "Who is the author of 'Pride and Prejudice'?",
        "Who is the author of 'To Kill a Mockingbird'?",
        "What is the chemical symbol for potassium?",
        "What is the chemical symbol for gold?",
        "What is the atomic number of carbon?",
        "In what year did the Titanic sink?",
        "How many players are on a standard soccer team on the field at one time?",
        "Translate 'apple' into French",
        "What does 'HTML' stand for?",
        "If all cats are animals, and all animals breathe, do all cats breathe?",
        "Who played Jack in the movie 'Titanic'?",
        "If a central bank raises interest rates significantly, what is likely to happen to borrowing and spending?",
        "Why did the stock market crash of 1929 lead to widespread unemployment?",
        "What happens to sea levels if polar ice caps melt?",
        "What is the likely outcome of administering antibiotics to a patient with a viral infection?",
        "If a company's servers are hacked and customer data is stolen, what are some potential consequences?",
        "What happens to public trust when government officials are caught in corruption scandals?",
        "If a car suddenly brakes on a wet road, what is likely to happen to its stopping distance compared to a dry road?",
        "How does economic inequality often influence political instability?",
        "What happens to crop yields during a severe drought?",
        "What happens to a country's energy costs if it shifts from fossil fuels to renewable energy sources in the short term?",
        "What is the likely outcome of a country imposing tariffs on imported goods?"
           ]

expected_answers = ["391",
                    "Paris",
                    "Canberra",
                    "16",
                    "8.4 million",
                    "Jupiter",
                    "Jane Austen",
                    "Harper Lee",
                    "K",
                    "Au",
                    "6",
                    "1912",
                    "11",
                    "pomme",
                    "HyperText Markup Language",
                    "Yes",
                    "Leonardo DiCaprio",
                    "Borrowing and spending are likely to decrease",
                    "Widespread unemployment is likely to occur because companies will have to lay off workers to cut costs",
                    "Sea levels will rise",
                    "The antibiotics will have no effect on the viral infection",
                    "Potential consequences include loss of customer trust, lawsuits, and financial losses",
                    "Public trust in the government is likely to decrease",
                    "The stopping distance will increase",
                    "Economic inequality often leads to political instability because it creates social unrest and dissatisfaction",
                    "Crop yields will decrease",
                    "Energy costs will likely increase in the short term",
                    "Tariffs raise import costs, protect domestic industries, and risk trade retaliation."
                    ]


def generate_response(prompt, temperature=0.7):

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate a response
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, 
                                 # attention_mask=inputs["attention_mask"],
                                 max_length=100, 
                                 temperature=temperature, 
                                 do_sample=True)

    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

temperatures = np.arange(0.2, 2.0, 0.3)
outputs = {}

for p, a in zip(prompts, expected_answers):
    prompt = system_prompt + p + assistant_prompt
    print("***********************************************************")
    print("Query:", p)
    print("Expected Answer:", a)
    print("***********************************************************")
    for temp in temperatures:
        temp = round(float(temp), 1)
        print("============================================================")
        print(f"Generating response with temperature: {temp}")
        response = generate_response(prompt, temperature=temp)
        outputs[temp] = response
        print(response)