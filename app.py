import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MT = "facebook/nllb-200-distilled-600M"
SRC, TGT = "eng_Latn", "hin_Deva"   # NLLB language codes
THRESHOLD = 0.88                    # similarity needed to reuse memory

print("Loading models (first run downloads ~2.5 GB)...")
embedder = SentenceTransformer(EMB)
tok = AutoTokenizer.from_pretrained(MT, src_lang=SRC)
model = AutoModelForSeq2SeqLM.from_pretrained(MT)

# seed translation memory (source, target)
memory = [
    ("Good morning", "सुप्रभात"),
    ("Thank you very much", "आपका बहुत बहुत धन्यवाद"),
    ("Where is the railway station?", "रेलवे स्टेशन कहाँ है?"),
]

dim = embedder.get_sentence_embedding_dimension()
index = faiss.IndexFlatIP(dim)


def embed(texts):
    return np.asarray(embedder.encode(texts, normalize_embeddings=True), dtype="float32")


index.add(embed([s for s, _ in memory]))


def nllb(text):
    inputs = tok(text, return_tensors="pt")
    out = model.generate(
        **inputs,
        forced_bos_token_id=tok.convert_tokens_to_ids(TGT),
        max_new_tokens=200,
    )
    return tok.batch_decode(out, skip_special_tokens=True)[0]


def translate(text):
    scores, ids = index.search(embed([text]), 1)
    score, i = float(scores[0][0]), int(ids[0][0])
    if score >= THRESHOLD:
        return memory[i][1], f"memory hit ({score:.2f})"
    result = nllb(text)
    memory.append((text, result))
    index.add(embed([text]))
    return result, "NLLB"


if __name__ == "__main__":
    print("Ready. Type English text (empty line to quit).")
    while True:
        q = input("English> ").strip()
        if not q:
            break
        out, src = translate(q)
        print(f"{out}   [{src}]")
