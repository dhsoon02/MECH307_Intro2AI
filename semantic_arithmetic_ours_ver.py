import torch
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_emb(word: str):
    return model.encode(word, convert_to_tensor=True, normalize_embeddings=True)

# semantic arithmetic: apple - phone + computer
a = get_emb("apple")
b = get_emb("phone")
c = get_emb("computer")

query = a - b + c
query = query / query.norm()

candidates = ["macbook", "ipad", "imac", "iphone", "laptop",
              "samsung", "microsoft", "windows", "google"]

cand_embs = model.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)
scores = util.cos_sim(query, cand_embs).squeeze()

print("apple - phone + computer ≈ ?")
for idx in scores.argsort(descending=True):
    print(f" {candidates[idx]}: (similarity: {scores[idx]:.4f})")
