import torch
from sentence_transformers import SentenceTransformer, util

# 1. Define the models to test
models_to_test = [
    "paraphrase-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "all-distilroberta-v1"
]

# 2. Function to extract embedding for a given word with a given model
def get_embedding(model, word: str) -> torch.Tensor:
    emb = model.encode(word, convert_to_tensor=True, normalize_embeddings=True)
    return emb

# 3. Candidate words
candidates = ["queen", "monarch", "princess", "royal", "empress",
              "dog", "car", "computer", "dgist", "aprl"]

# 4. Loop through each model and perform semantic arithmetic
for model_name in models_to_test:
    print("="*70)
    print(f"Model: {model_name}")
    model = SentenceTransformer(model_name)

    # Perform king - man + woman
    king = get_embedding(model, "king")
    man = get_embedding(model, "man")
    woman = get_embedding(model, "woman")

    query = king - man + woman
    query = query / query.norm()  # normalize

    # Compute embeddings for candidates
    candidate_embeddings = model.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(query, candidate_embeddings).squeeze()

    # Print results in descending order
    print("king - man + woman ≈ ?")
    for idx in scores.argsort(descending=True):
        print(f" {candidates[idx]}: (similarity: {scores[idx]:.4f})")
