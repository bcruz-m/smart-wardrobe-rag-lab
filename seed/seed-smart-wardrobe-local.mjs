#!/usr/bin/env node
/**
 * M4.3.1 (free variant) — embed the existing Smart Wardrobe chunks locally with
 * all-MiniLM-L6-v2 (384d) and upsert them to a dedicated 384d Pinecone index.
 *
 * Reads data/smart-wardrobe/smart-wardrobe-chunks.json (never rewrites it).
 * No OpenAI / paid API calls.
 *
 * Usage:
 *   npm run seed:smart-wardrobe:local -- --dry-run   # embed + check dims only, no Pinecone
 *   npm run seed:smart-wardrobe:local                # create index if missing, upsert, verify
 */

import { readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  embedLocal,
  LOCAL_EMBEDDING_DIM,
  LOCAL_EMBEDDING_MODEL,
} from "../example/shared/local-embedder.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

// Fixed target — intentionally NOT read from env, so there is no shared-namespace fallback.
const INDEX_NAME = "bootcamp-m43-minilm";
const NAMESPACE = "bruna-m43";
const PROTECTED_INDEXES = ["bootcamp-m43"]; // 1536d OpenAI index — never write 384d vectors there
const FORBIDDEN_NAMESPACES = ["workshop-shared"];
const METRIC = "cosine";
const CLOUD = "aws";
const REGION = "us-east-1"; // only region on the Starter plan
const STARTER_MAX_INDEXES = 5;
const CHUNKS_JSON = "data/smart-wardrobe/smart-wardrobe-chunks.json";

function loadEnvLocal() {
  for (const name of [".env.local", ".env"]) {
    const path = join(ROOT, name);
    if (!existsSync(path)) continue;
    for (const line of readFileSync(path, "utf8").split("\n")) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const eq = trimmed.indexOf("=");
      if (eq === -1) continue;
      const key = trimmed.slice(0, eq).trim();
      let val = trimmed.slice(eq + 1).trim();
      if (
        (val.startsWith('"') && val.endsWith('"')) ||
        (val.startsWith("'") && val.endsWith("'"))
      ) {
        val = val.slice(1, -1);
      }
      if (process.env[key] === undefined || process.env[key] === "") {
        process.env[key] = val;
      }
    }
  }
}

async function ensureIndex(pc) {
  const indexes = (await pc.listIndexes()).indexes ?? [];
  const existing = indexes.find((i) => i.name === INDEX_NAME);
  if (existing) {
    if (existing.dimension !== LOCAL_EMBEDDING_DIM || existing.metric !== METRIC) {
      throw new Error(
        `Index "${INDEX_NAME}" is ${existing.dimension}d/${existing.metric}, expected ${LOCAL_EMBEDDING_DIM}d/${METRIC}`,
      );
    }
    console.log(`Using existing index "${INDEX_NAME}" (${existing.dimension}d, ${existing.metric})`);
    return;
  }
  if (indexes.length >= STARTER_MAX_INDEXES) {
    throw new Error(
      `Project already has ${indexes.length} indexes (Starter limit ${STARTER_MAX_INDEXES}) — not creating "${INDEX_NAME}"`,
    );
  }
  console.log(
    `Creating index "${INDEX_NAME}" (${LOCAL_EMBEDDING_DIM}d, ${METRIC}, ${CLOUD}/${REGION}) — ` +
      `${indexes.length + 1}/${STARTER_MAX_INDEXES} indexes after creation…`,
  );
  await pc.createIndex({
    name: INDEX_NAME,
    dimension: LOCAL_EMBEDDING_DIM,
    metric: METRIC,
    spec: { serverless: { cloud: CLOUD, region: REGION } },
    waitUntilReady: true,
  });
  console.log(`✓ Index "${INDEX_NAME}" ready`);
}

async function main() {
  const dryRun = process.argv.includes("--dry-run");

  if (PROTECTED_INDEXES.includes(INDEX_NAME)) {
    throw new Error(`Refusing to write ${LOCAL_EMBEDDING_DIM}d vectors to protected index "${INDEX_NAME}"`);
  }
  if (FORBIDDEN_NAMESPACES.includes(NAMESPACE)) {
    throw new Error(`Refusing to write to shared namespace "${NAMESPACE}"`);
  }

  const { chunks } = JSON.parse(readFileSync(join(ROOT, CHUNKS_JSON), "utf8"));
  console.log(`Loaded ${chunks.length} chunks from ${CHUNKS_JSON}`);

  console.log(`Embedding locally with ${LOCAL_EMBEDDING_MODEL}…`);
  const started = Date.now();
  const embeddings = await embedLocal(chunks.map((c) => c.chunk_text));
  console.log(
    `✓ ${embeddings.length} vectors × ${embeddings[0].length} dims in ${Date.now() - started} ms`,
  );

  if (dryRun) {
    console.log("Dry run — no Pinecone calls made.");
    return;
  }

  loadEnvLocal();
  const pineconeKey = process.env.PINECONE_API_KEY;
  if (!pineconeKey) throw new Error("Missing PINECONE_API_KEY — set it in .env or .env.local");

  const { Pinecone } = await import("@pinecone-database/pinecone");
  const pc = new Pinecone({ apiKey: pineconeKey });

  await ensureIndex(pc);
  const desc = await pc.describeIndex(INDEX_NAME);
  const index = pc.index(INDEX_NAME, desc.host);
  const ns = index.namespace(NAMESPACE);

  const vectors = chunks.map(({ id, ...metadata }, i) => ({
    id,
    values: embeddings[i],
    metadata: { ...metadata, embedding_model: LOCAL_EMBEDDING_MODEL },
  }));
  console.log(`Upserting ${vectors.length} vectors to ${INDEX_NAME}/${NAMESPACE}…`);
  await ns.upsert({ records: vectors });

  const fetched = await ns.fetch({ ids: chunks.map((c) => c.id) });
  const records = fetched.records ?? {};
  const missing = chunks.filter((c) => records[c.id]?.values?.length !== LOCAL_EMBEDDING_DIM);
  if (missing.length) {
    throw new Error(`Upsert verification failed: ${missing.map((c) => c.id).join(", ")}`);
  }
  console.log(`✓ Verified ${chunks.length}/${chunks.length} vectors by fetch (${LOCAL_EMBEDDING_DIM}d each)`);

  // Stats are eventually consistent — poll briefly until the count catches up.
  let count = 0;
  for (let attempt = 0; attempt < 10; attempt++) {
    const stats = await index.describeIndexStats();
    count = stats.namespaces?.[NAMESPACE]?.recordCount ?? 0;
    if (count >= chunks.length) break;
    await new Promise((r) => setTimeout(r, 2000));
  }
  console.log(`  Namespace "${NAMESPACE}" record count: ${count}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
