/**
 * Free, local sentence embeddings via Transformers.js (ONNX, runs on CPU).
 * Model: all-MiniLM-L6-v2 → 384-dimensional, L2-normalised vectors.
 * The model (~90 MB) is downloaded once from huggingface.co into ./.cache/models.
 */

import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "../..");

export const LOCAL_EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
export const LOCAL_EMBEDDING_DIM = 384;

let extractorPromise;

async function getExtractor() {
  if (!extractorPromise) {
    extractorPromise = (async () => {
      const { pipeline, env } = await import("@huggingface/transformers");
      env.cacheDir = join(ROOT, ".cache/models");
      return pipeline("feature-extraction", LOCAL_EMBEDDING_MODEL, { dtype: "fp32" });
    })();
  }
  return extractorPromise;
}

/** Embed texts locally; throws if any vector is not LOCAL_EMBEDDING_DIM long. */
export async function embedLocal(texts) {
  const extractor = await getExtractor();
  const output = await extractor(texts, { pooling: "mean", normalize: true });
  const vectors = output.tolist();
  const bad = vectors.findIndex((v) => v.length !== LOCAL_EMBEDDING_DIM);
  if (bad !== -1) {
    throw new Error(
      `Embedding ${bad} has ${vectors[bad].length} dims, expected ${LOCAL_EMBEDDING_DIM}`,
    );
  }
  return vectors;
}
