#!/usr/bin/env node
/**
 * M4.3.1 — Extract the Smart Wardrobe PDFs and chunk them by section heading.
 * Writes data/smart-wardrobe/smart-wardrobe-chunks.json. No API calls.
 *
 * Embedding + upload is done separately by seed/seed-smart-wardrobe-local.mjs
 * (local all-MiniLM-L6-v2, 384d); the JSON header records that configuration.
 *
 * Usage:
 *   npm run extract:smart-wardrobe
 */

import { readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { LOCAL_EMBEDDING_MODEL } from "../example/shared/local-embedder.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

// Recorded in the JSON header; must match seed/seed-smart-wardrobe-local.mjs.
const INDEX_NAME = "bootcamp-m43-minilm";
const NAMESPACE = "bruna-m43";
const MAX_CHUNK_CHARS = 1200;

const SOURCES = [
  { file: "data/smart-wardrobe/Smart_Wardrobe_Features_and_User_Guide.pdf", idPrefix: "features-guide" },
  { file: "data/smart-wardrobe/Smart_Wardrobe_Technical_Overview.pdf", idPrefix: "technical-overview" },
];
const OUTPUT_JSON = "data/smart-wardrobe/smart-wardrobe-chunks.json";

// "Document purpose" or numbered section lines like "7. User corrections and trust".
const HEADING_RE = /^(Document purpose|\d{1,2}\.\s+[A-Z].{0,80})$/;

async function extractPages(relPath) {
  const { extractText, getDocumentProxy } = await import("unpdf");
  const pdf = await getDocumentProxy(new Uint8Array(readFileSync(join(ROOT, relPath))));
  const { text } = await extractText(pdf, { mergePages: false });
  return text;
}

/** Split an over-long body at sentence boundaries into pieces <= MAX_CHUNK_CHARS. */
function splitLong(body) {
  if (body.length <= MAX_CHUNK_CHARS) return [body];
  const sentences = body.match(/[^.!?]+[.!?]+(\s|$)|[^.!?]+$/g) ?? [body];
  const parts = [];
  let current = "";
  for (const s of sentences) {
    if (current && (current + s).length > MAX_CHUNK_CHARS) {
      parts.push(current.trim());
      current = "";
    }
    current += s;
  }
  if (current.trim()) parts.push(current.trim());
  return parts;
}

function chunkDocument(pages, source) {
  const lines = pages.flatMap((text, i) =>
    text.split("\n").map((l) => ({ text: l.trim(), page: i + 1 })).filter((l) => l.text),
  );
  if (lines.length < 2) throw new Error(`No text extracted from ${source.file}`);

  // Line 1 = document title, line 2 = subtitle; sections start after.
  const documentTitle = lines[0].text;
  const sections = [];
  for (const line of lines.slice(2)) {
    if (HEADING_RE.test(line.text)) {
      sections.push({ heading: line.text, page: line.page, body: [] });
    } else if (sections.length) {
      sections.at(-1).body.push(line.text);
    } else {
      throw new Error(`Text before first heading in ${source.file}: "${line.text}"`);
    }
  }

  const chunks = [];
  for (const section of sections) {
    const body = section.body.join(" ").replace(/\s+/g, " ").trim();
    if (!body) continue;
    for (const piece of splitLong(body)) {
      const chunkNumber = chunks.length + 1;
      chunks.push({
        id: `${source.idPrefix}-${String(chunkNumber).padStart(3, "0")}`,
        heading: section.heading,
        source_file: basename(source.file),
        chunk_number: chunkNumber,
        document_title: documentTitle,
        page: section.page,
        chunk_text: `${documentTitle} — ${section.heading}\n${piece}`,
      });
    }
  }
  return chunks;
}

async function main() {
  // 1. Extract + chunk
  const chunks = [];
  for (const source of SOURCES) {
    const pages = await extractPages(source.file);
    const docChunks = chunkDocument(pages, source);
    console.log(`Extracted ${basename(source.file)}: ${pages.length} pages → ${docChunks.length} chunks`);
    chunks.push(...docChunks);
  }

  // 2. Save local JSON
  writeFileSync(
    join(ROOT, OUTPUT_JSON),
    JSON.stringify(
      {
        generated_at: new Date().toISOString(),
        embedding_model: LOCAL_EMBEDDING_MODEL,
        index: INDEX_NAME,
        namespace: NAMESPACE,
        sources: SOURCES.map((s) => basename(s.file)),
        total_chunks: chunks.length,
        chunks,
      },
      null,
      2,
    ) + "\n",
  );
  console.log(`✓ Wrote ${chunks.length} chunks to ${OUTPUT_JSON}`);
  console.log("Next: npm run seed:smart-wardrobe:local  (local embeddings + Pinecone upload)");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
