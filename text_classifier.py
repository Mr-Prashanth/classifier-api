from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

def score(text):
    return nlp(text)[0]["score"]

def classify(text, threshold = 0.90):
    result = score(text)

    if result >= threshold:
        return "it is not ok..."
    return "it is ok..."

# Load the tokenizer and model
model_name = "unitary/unbiased-toxic-roberta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create a pipeline for text classification
nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)


# Example texts

if __name__ == "__main__":
    texts = [
        "You are such a loser, no one likes you!",
        "I hope you have a great day!",
        "You should stop posting these things, it's really dumb."
    ]

    # Run the classification
    for text in texts:
        result = nlp(text)
        print(f"Text: {text}\nClassification: {result}\n")