# Smart Wardrobe RAG Lab (M4.3)

Exercise M4.3 of the Digitale Leute Vibe Coding Bootcamp. Two Smart Wardrobe PDFs are split into chunks with metadata. The chunks are embedded locally and stored in Pinecone, and six questions are answered by a deterministic RAG chain that uses a local LLM. **No paid APIs are required.**

Results and the question table are in [`M4.3-Submission.md`](M4.3-Submission.md).

## How it works

```
PDFs ──unpdf──► 22 section chunks (+ metadata) ──all-MiniLM-L6-v2 (local, 384d)──► Pinecone
                                                                                      │
question ──MiniLM──► top-3 retrieval ──► qwen2.5:3b via Ollama (local, temp 0, seed 42)
                                          │
                     code verifies quoted evidence → code-built citations → grounding checks
                     (failed check → verified source quotes, labelled as rule-based fallback)
```

| Component | Choice |
|---|---|
| PDF text extraction | `unpdf` |
| Embeddings | `Xenova/all-MiniLM-L6-v2` via `@huggingface/transformers` (local, 384 dimensions) |
| Vector store | Pinecone serverless index `bootcamp-m43-minilm` (384d, cosine, aws/us-east-1), namespace `bruna-m43` |
| LLM | `qwen2.5:3b` via [Ollama](https://ollama.com) (local) |

This adapts the bootcamp's workshop RAG example, which used OpenAI embeddings and `gpt-4o-mini`. Here, free local models are used instead, and stricter grounding is added: evidence must be quoted verbatim, citations come from metadata, and refusals are explicit.

## Project structure

```
example/rag/smart-wardrobe-local-rag.mjs   deterministic RAG chain (M4.3.2), writes the report
example/shared/local-embedder.mjs          local MiniLM embedder (shared)
seed/seed-smart-wardrobe-local.mjs         embeds chunks locally and upserts to Pinecone (M4.3.1)
seed/seed-smart-wardrobe-vectors.mjs       PDF extraction + chunking → smart-wardrobe-chunks.json (no API calls)
data/smart-wardrobe/*.pdf                  the two source documents
data/smart-wardrobe/smart-wardrobe-chunks.json   22 extracted chunks with metadata
data/smart-wardrobe/m43-2-report.{md,json}       question/answer/citation report
M4.3-Submission.md                         exercise submission
```

## Requirements

- Node.js 20 or newer (developed with Node 24)
- A free [Pinecone](https://www.pinecone.io) account and API key. The Starter plan is enough, but it is limited to 5 indexes and the aws/us-east-1 region.
- [Ollama](https://ollama.com), only for the question-answering step
- Disk space: about 100 MB for the embedding model and about 1.9 GB for `qwen2.5:3b`

## Setup

```bash
npm install
cp env.example .env          # then set PINECONE_API_KEY in .env (never commit it)

brew install ollama          # macOS; see ollama.com for other platforms
ollama serve                 # keep running in a separate terminal
ollama pull qwen2.5:3b
```

The MiniLM model (about 90 MB) is downloaded from Hugging Face on first use and cached in `./.cache/models`.

## Run the exercise

```bash
# M4.3.1: embed the 22 chunks locally and upload them to Pinecone.
# Creates the 384d index if it is missing, then verifies by fetch and checks the namespace count.
npm run seed:smart-wardrobe:local -- --dry-run   # local embedding + dimension check only
npm run seed:smart-wardrobe:local

# M4.3.2: answer the six questions and regenerate data/smart-wardrobe/m43-2-report.{md,json}
npm run example:rag:smart-wardrobe
npm run example:rag:smart-wardrobe -- --retrieve-only        # similarity scores only, no LLM
npm run example:rag:smart-wardrobe -- "Your own question?"   # one ad-hoc question, no report
```

Re-running the seed script overwrites the same 22 vector IDs, so it never creates duplicates. The RAG chain is deterministic: `temperature 0`, a fixed seed, and the model is unloaded after each call (`keep_alive: 0`).

### Re-extracting chunks from the PDFs (optional)

`smart-wardrobe-chunks.json` is already included. To regenerate it from the PDFs:

```bash
npm run extract:smart-wardrobe
```

This only extracts and chunks the PDFs, and makes no API calls. It writes the header with the local configuration (`Xenova/all-MiniLM-L6-v2`, `bootcamp-m43-minilm`, `bruna-m43`). Only `generated_at` changes. Afterwards, run `npm run seed:smart-wardrobe:local` to embed and upload the chunks.

## Known limitations

- MiniLM is a small English-only model. Retrieval can miss the best passage. For example, Q2 does not retrieve the "users should be able to correct incorrect suggestions" sentence from Features §2.
- The grounding checks are lexical heuristics (word overlap and hedge words), not semantic verification.
- A 3B model sometimes overstates uncertain facts. When that happens, the chain returns the verified source quotes instead, and labels them as a rule-based fallback.
- Determinism holds for the same machine, Ollama version and model version.
