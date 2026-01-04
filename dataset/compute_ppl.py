import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Modelo zero-shot para calcular PPL
MODEL_NAME = "gpt2"

def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def compute_perplexity(text: str, tokenizer, model, device) -> float:
    if not isinstance(text, str) or text.strip() == "":
        return float("nan")

    # Tokenización
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    input_ids = encodings["input_ids"].to(device)

    with torch.no_grad():
        # Para modelos CausalLM, podemos usar labels = input_ids
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # cross-entropy media sobre todos los tokens
        perplexity = torch.exp(loss).item()

    return perplexity


def main():
    # Cargar dataset
    dataset_path = 'dataset_filter3_ngram.csv'
    df = pd.read_csv(dataset_path)

    tokenizer, model, device = load_model_and_tokenizer(MODEL_NAME)

    perplexities = []
    total = len(df)
    print("Calculando perplexity por comentario...")
	
    for text in tqdm(df['text'], total=total, desc="Perplexity", unit="comentario"):
        ppl = compute_perplexity(text, tokenizer, model, device)
        perplexities.append(ppl)

    df["perplexity"] = perplexities

    df.to_csv('dataset_filter3_ppl.csv', index=False)
    print("Listo")

if __name__ == "__main__":
    main()

