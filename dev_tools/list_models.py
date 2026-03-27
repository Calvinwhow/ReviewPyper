import openai
api_key_path = "/Users/rm026/Documents/code/openai-key.txt"
with open(api_key_path, 'r') as f:
    key = f.read().strip()
client = openai.OpenAI(api_key=key)
print("Isolating... attempting API list...")
models=client.models.list()
for model in models:
    print(model,'\n')

print("API Reachable!")