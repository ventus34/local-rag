from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import json

# List of models to download
model_names = [
    'jinaai/jina-embeddings-v2-base-code',  # ~560MB - Code retrieval
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',  # ~1.1GB - Multilingual docs (large)
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',  # ~471MB - Multilingual docs (medium)
    'sentence-transformers/all-MiniLM-L6-v2',  # ~90MB - Very lightweight universal model
    'cross-encoder/ms-marco-MiniLM-L6-v2'  # ~90MB - reranker
]

# Create the target directory if it doesn't exist
output_folder = "./models"
os.makedirs(output_folder, exist_ok=True)

# Dictionary to track downloaded models and their local paths
downloaded_models = {}

# Loop to download and save each model
for model_name in model_names:
    print(f"Downloading {model_name}...")
    # The folder name will be a safe version of the model name
    safe_name = model_name.replace('/', '_')
    save_path = f"{output_folder}/{safe_name}"

    if os.path.exists(save_path):
        print(f"Model already exists at {save_path}. Skipping.")
        downloaded_models[model_name] = save_path
        continue

    try:
        # Select the appropriate class depending on the model type
        if 'reranker' in model_name.lower():
            model = CrossEncoder(model_name)
        else:
            model = SentenceTransformer(model_name)

        model.save(save_path)
        print(f"Model saved to {save_path}")
        downloaded_models[model_name] = save_path

    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        continue

# Create a model mapping file for easy reference
model_mapping = {
    "model_paths": downloaded_models,
    "output_folder": output_folder
}

mapping_file = os.path.join(output_folder, "model_mapping.json")
with open(mapping_file, 'w') as f:
    json.dump(model_mapping, f, indent=2)

print(f"\nModel mapping saved to {mapping_file}")
print(f"\nAll models downloaded successfully to {output_folder}!")
print("\nDownloaded models:")
for original_name, local_path in downloaded_models.items():
    print(f"  {original_name} -> {local_path}")