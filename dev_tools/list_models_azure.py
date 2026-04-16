import openai

api_key_path = "F:/Code/openai-key.txt"
with open(api_key_path, 'r') as f:
    key = f.read().strip()

api_base = "https://mgb-risc-wrkspce-prod-e2-8-cog.openai.azure.com/"
api_version = "2025-01-01-preview"
client = openai.AzureOpenAI(
                api_key=key,
                api_version=api_version,
                azure_endpoint=api_base
            )

test_question='Respond "Hello World!"'
deployment='gpt-4.1-mini'

response = client.chat.completions.create(
    model=deployment,
    messages=[
        {"role": "user", "content": test_question}
    ],
        max_tokens=10  # Adjust as needed
    )
print(response.choices[0].message.content.strip())
# print("Isolating... attempting API list...")
# models=client.models.list()
# for model in models:
#     print(model,'\n')

print("API Reachable!")