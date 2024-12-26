import random
import json
import torch
from nltk_utils import tokenize, bag_of_words
from model import NeuralNet

# Load the intents and model data
with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "chatbot_model.pth"
data = torch.load(FILE, weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bot_name = "Mango"
print(f"{bot_name}: Hello! Type 'quit' to exit the chat.")

while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        print(f"{bot_name}: Goodbye!")
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).to(device)

    output = model(X.unsqueeze(0))
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]


    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand. Can you please rephrase?")
