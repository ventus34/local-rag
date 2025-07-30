from sentence_transformers import SentenceTransformer, CrossEncoder
import os

# List of models to download
model_names = [
    'BAAI/bge-m3',
    'jinaai/jina-embeddings-v2-base-code',
    'BAAI/bge-reranker-large'
]

# Create the target directory if it doesn't exist
output_folder = "./models"
os.makedirs(output_folder, exist_ok=True)


# Loop to download and save each model
for model_name in model_names:
    print(f"Downloading {model_name}...")
    # The folder name will be a safe version of the model name
    save_path = f"{output_folder}/{model_name.replace('/', '_')}"

    if os.path.exists(save_path):
        print(f"Model already exists at {save_path}. Skipping.")
        continue

    # Select the appropriate class depending on the model type
    if 'reranker' in model_name:
        model = CrossEncoder(model_name)
    else:
        model = SentenceTransformer(model_name)

    model.save(save_path)
    print(f"Model saved to {save_path}")

print("\nAll models downloaded successfully!")