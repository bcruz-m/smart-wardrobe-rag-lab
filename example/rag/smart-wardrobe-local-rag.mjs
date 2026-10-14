#!/usr/bin/env node
/**
 * M4.3.2 — Deterministic RAG over the Smart Wardrobe PDFs, fully local / free.
 *
 * Fixed pipeline for every question (the model never decides whether to search):
 *   1. embed question locally (all-MiniLM-L6-v2, 384d)
 *   2. retrieve top-K chunks from Pinecone (bootcamp-m43-minilm / bruna-m43)
 *   3. score gate: best score < MIN_SCORE → refuse without calling the LLM
 *   4. local LLM (Ollama qwen2.5:3b, temperature 0, fixed seed) returns JSON:
 *      verbatim evidence sentences + an answer built only from them
 *   5. code verifies every evidence quote is literally present in a retrieved chunk;
 *      no verified evidence → refusal
 *   6. citations = the chunks that contain verified evidence (built in code, never by the model)
 *   7. grounding checks: hedges in the evidence ("candidate", "does not assert", …) must survive
 *      in the answer, and the answer may not introduce content words absent from the evidence.
 *      A failed check replaces the LLM answer with the verified evidence quotes (labelled).
 *
 * Usage:
 *   npm run example:rag:smart-wardrobe                     # six report questions → report files
 *   npm run example:rag:smart-wardrobe -- --retrieve-only  # scores only, no LLM, no report
 *   npm run example:rag:smart-wardrobe -- "Your question?" # single ad-hoc question
 *
 * Requires `ollama serve` running locally with `qwen2.5:3b` pulled.
 */

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { embedLocal, LOCAL_EMBEDDING_MODEL } from "../shared/local-embedder.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "../..");

const INDEX_NAME = "bootcamp-m43-minilm";
const NAMESPACE = "bruna-m43";
const TOP_K = 3;
// Cosine floor for all-MiniLM-L6-v2. Every chunk starts with "Smart Wardrobe — …", so on-topic but
// unsupported questions still score ~0.45; this gate only blocks clearly unrelated questions and
// the evidence check handles the rest (calibrated with --retrieve-only).
const MIN_SCORE = 0.35;
const OLLAMA_URL = "http://localhost:11434";
const LLM_MODEL = "qwen2.5:3b";
const LLM_OPTIONS = { temperature: 0, seed: 42, top_k: 1, num_ctx: 4096 };
const REFUSAL = "This information is not available in the Smart Wardrobe documents.";
const REPORT_BASE = "data/smart-wardrobe/m43-2-report";
const MIN_QUOTE_CHARS = 20;

// Words that mark uncertainty / non-final status. If the evidence has one, the answer must too.
const HEDGES = [
  "candidate", "proposed", "planned", "future", "concept", "suggested", "example", "sample",
  "may", "could", "would", "does not assert", "not confirmed", "not established", "not a claim",
  "not implied",
];
// Evidence must share at least this many content words with the final answer to be cited.
const MIN_SUPPORT_OVERLAP = 2;
const STOPWORDS = new Set(
  ("the and for with that this these those from into about also only both than then there their " +
    "they them such which what when where does have has had were was are can should would could " +
    "may not but any all each its it's user users smart wardrobe app application document documents " +
    "described describes states state stated according passage passages context " +
    "include includes included including contain contains contained consist consists").split(" "),
);

const QUESTIONS = [
  { question: "What clothing categories does Smart Wardrobe use?", expect: "answer" },
  { question: "What should a user do if the AI classifies a clothing item incorrectly?", expect: "answer" },
  { question: "Which fields are in the proposed clothing item schema?", expect: "answer" },
  { question: "Is Supabase currently connected to the Smart Wardrobe app?", expect: "answer" },
  { question: "How should virtual try-on be designed so the provider can be replaced?", expect: "answer" },
  { question: "How much does the Smart Wardrobe premium subscription cost per month?", expect: "refuse" },
];

const SYSTEM_PROMPT = `You answer questions about the Smart Wardrobe product using ONLY the numbered context passages.
Return JSON with two fields:
- "evidence": 1-3 sentences copied WORD FOR WORD from the passage text that directly answer the question. Copy only body sentences, never the passage title line. Do not shorten, merge or reword them.
- "answer": 1-2 sentences that answer the question using ONLY what the evidence sentences say.
Rules for "answer":
- Do not add facts, reasons or conclusions that are not in the evidence.
- Keep the evidence's level of certainty. If the evidence says something is a candidate, proposed, planned, suggested, or that the document "does not assert" something, say exactly that. Never turn it into a definite yes or no.
- Do not include passage numbers or citations, and no framing phrases such as "based on the information provided".
If no passage directly answers the question, return {"evidence": [], "answer": "${REFUSAL}"}.`;

const RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    evidence: { type: "array", items: { type: "string" } },
    answer: { type: "string" },
  },
  required: ["evidence", "answer"],
};

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

async function getNamespace() {
  const apiKey = process.env.PINECONE_API_KEY;
  if (!apiKey) throw new Error("Missing PINECONE_API_KEY — set it in .env or .env.local");
  const { Pinecone } = await import("@pinecone-database/pinecone");
  const pc = new Pinecone({ apiKey });
  const host = (await pc.describeIndex(INDEX_NAME)).host;
  return pc.index(INDEX_NAME, host).namespace(NAMESPACE);
}

async function retrieve(ns, question) {
  const [vector] = await embedLocal([question]);
  const results = await ns.query({ vector, topK: TOP_K, includeMetadata: true });
  return (results.matches ?? []).map((m) => ({
    id: m.id,
    score: Number(m.score.toFixed(4)),
    document_title: m.metadata.document_title,
    heading: m.metadata.heading,
    source_file: m.metadata.source_file,
    page: m.metadata.page,
    chunk_text: m.metadata.chunk_text,
    // chunk_text = "<title> — <heading>\n<body>"; quotes are verified against the body only
    body: m.metadata.chunk_text.split("\n").slice(1).join(" "),
  }));
}

function citationOf(hit) {
  return `${hit.document_title} — ${hit.heading} (${hit.source_file}, p.${hit.page})`;
}

const normalize = (s) =>
  s.toLowerCase().replace(/[‘’]/g, "'").replace(/[“”]/g, '"').replace(/\s+/g, " ").trim();

const splitSentences = (text) =>
  text.split(/(?<=[.!?])\s+/).map((t) => t.trim()).filter(Boolean);

/**
 * Verified evidence = sentences of the retrieved chunk bodies that the model actually quoted.
 * A source sentence counts if it occurs verbatim inside a quote (tolerates wrapping/headers the
 * model may add) or if a quote of >= MIN_QUOTE_CHARS is a verbatim fragment of it.
 * The stored quote is always the document's own sentence, never the model's text.
 */
function verifyEvidence(evidence, hits) {
  const quotes = evidence.map((q) => normalize(q).replace(/[.\s]+$/, "")).filter(Boolean);
  const used = new Set();
  const verified = [];
  for (const hit of hits) {
    for (const sentence of splitSentences(hit.body)) {
      const ns = normalize(sentence).replace(/[.\s]+$/, "");
      quotes.forEach((q, qi) => {
        const match = q.includes(ns) || (q.length >= MIN_QUOTE_CHARS && ns.includes(q));
        if (!match) return;
        used.add(qi);
        if (!verified.some((v) => v.quote === sentence)) verified.push({ quote: sentence, source: hit });
      });
    }
  }
  const rejected = evidence.filter((_, i) => !used.has(i) && quotes[i]);
  return { verified, rejected };
}

// Refusals phrased in the model's own words still count as refusals.
const REFUSAL_PATTERNS = [
  /not available in the smart wardrobe documents/i,
  /\b(is|are) not (mentioned|provided|specified|stated|included|available)\b/i,
  /\bdo(es)? not (mention|provide|specify|state|include|contain)\b/i,
  /\bno (information|mention|details?)\b/i,
];
const isRefusal = (text) => REFUSAL_PATTERNS.some((re) => re.test(text));

const hasHedge = (text) => HEDGES.some((h) => new RegExp(`\\b${h}\\b`, "i").test(text));
const contentWords = (text) =>
  (normalize(text).match(/[a-z_][a-z_'-]{3,}/g) ?? []).filter((w) => !STOPWORDS.has(w));
const stem = (w) =>
  w
    .replace(/'s$/, "")
    .replace(/(.{4,})ly$/, "$1")
    .replace(/(ing|ed|es|s)$/, "")
    .replace(/([b-df-hj-np-tv-z])\1$/, "$1");

function groundingChecks(answerText, verified, question) {
  const evidenceText = verified.map((v) => v.quote).join(" ");
  const known = new Set([...contentWords(evidenceText), ...contentWords(question)].map(stem));
  const unsupported_terms = [...new Set(contentWords(answerText).filter((w) => !known.has(stem(w))))];
  // A hedge is required only if it sits in an evidence sentence about the same thing as the answer
  // (shares a content word), e.g. "may misclassify" does not bind an answer about reviewing labels.
  const answerStems = new Set(contentWords(answerText).map(stem));
  const relatedText = verified
    .map((v) => v.quote)
    .filter((q) => contentWords(q).some((w) => answerStems.has(stem(w))))
    .join(" ");
  const evidence_hedges = HEDGES.filter((h) => new RegExp(`\\b${h}\\b`, "i").test(relatedText));
  const hedge_dropped = evidence_hedges.length > 0 && !hasHedge(answerText);
  return {
    unsupported_terms,
    hedge_dropped,
    evidence_hedges,
    passed: !unsupported_terms.length && !hedge_dropped,
  };
}

function repairInstruction(checks) {
  const problems = [];
  if (checks.unsupported_terms.length) {
    problems.push(
      `it uses words that are not in the evidence (${checks.unsupported_terms.join(", ")}) — use the evidence's own wording instead`,
    );
  }
  if (checks.hedge_dropped) {
    problems.push(
      `it drops the evidence's qualifier (${checks.evidence_hedges.join(", ")}) — keep that qualifier in the answer`,
    );
  }
  return `Your answer is not strictly grounded: ${problems.join("; ")}. Return the same JSON with the same evidence and a corrected answer.`;
}

/** Evidence the final answer actually draws on (content-word overlap), used for citations. */
function supportingEvidence(answerText, verified) {
  const answerStems = new Set(contentWords(answerText).map(stem));
  const scored = verified.map((v) => ({
    ...v,
    overlap: new Set(contentWords(v.quote).map(stem).filter((w) => answerStems.has(w))).size,
  }));
  const used = scored.filter((v) => v.overlap >= MIN_SUPPORT_OVERLAP);
  return used.length ? used : [scored.sort((a, b) => b.overlap - a.overlap)[0]];
}

async function assertOllamaReady() {
  let tags;
  try {
    tags = await (await fetch(`${OLLAMA_URL}/api/tags`)).json();
  } catch {
    throw new Error(`Ollama is not reachable at ${OLLAMA_URL} — start it with: ollama serve`);
  }
  if (!tags.models?.some((m) => m.name === LLM_MODEL)) {
    throw new Error(`Model ${LLM_MODEL} not found — run: ollama pull ${LLM_MODEL}`);
  }
}

async function generate(question, context, repair) {
  const res = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: LLM_MODEL,
      stream: false,
      format: RESPONSE_SCHEMA,
      options: LLM_OPTIONS,
      // Unload after each call: reusing Ollama's prompt/KV cache across calls changes the float
      // math enough to flip temperature-0 outputs between runs. A cold call is reproducible.
      keep_alive: 0,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: `Context:\n${context}\n\nQuestion: ${question}` },
        ...(repair
          ? [
              { role: "assistant", content: JSON.stringify(repair.previous) },
              { role: "user", content: repair.instruction },
            ]
          : []),
      ],
    }),
  });
  if (!res.ok) throw new Error(`Ollama error ${res.status}: ${await res.text()}`);
  const content = (await res.json()).message.content;
  try {
    const parsed = JSON.parse(content);
    return {
      evidence: Array.isArray(parsed.evidence) ? parsed.evidence.map(String) : [],
      answer: String(parsed.answer ?? "").replace(/\s*\[\d+\]/g, "").trim(),
    };
  } catch {
    return { evidence: [], answer: "", parse_error: content };
  }
}

function uniqueCitations(verified) {
  const seen = new Map();
  for (const { source } of verified) if (!seen.has(source.id)) seen.set(source.id, source);
  return [...seen.values()].map((h, i) => ({ ref: i + 1, ...pick(h) }));
}

async function answer(ns, question) {
  const hits = await retrieve(ns, question);
  const usable = hits.filter((h) => h.score >= MIN_SCORE);
  const base = { question, top_score: hits[0]?.score ?? null, retrieved: hits.map(pick) };

  if (!usable.length) {
    return { ...base, mode: "REFUSAL (score gate, no LLM call)", answer: REFUSAL, evidence: [], citations: [] };
  }

  const context = usable
    .map((h, i) => `[${i + 1}] ${h.document_title} — ${h.heading}\n${h.body}`)
    .join("\n\n");
  const llm = await generate(question, context);
  const { verified, rejected } = verifyEvidence(llm.evidence, usable);
  const llmTrace = { llm_answer: llm.answer, llm_evidence: llm.evidence, rejected_evidence: rejected };

  if (isRefusal(llm.answer) || !verified.length) {
    const why = isRefusal(llm.answer) ? `LLM: ${LLM_MODEL}` : "no verifiable evidence";
    return { ...base, ...llmTrace, mode: `REFUSAL (${why})`, answer: REFUSAL, evidence: [], citations: [] };
  }

  let finalAnswer = llm.answer;
  let checks = groundingChecks(finalAnswer, verified, question);
  let repair = null;
  if (!checks.passed) {
    // One deterministic repair turn that names the exact grounding problem.
    const instruction = repairInstruction(checks);
    const repaired = await generate(question, context, {
      previous: { evidence: llm.evidence, answer: llm.answer },
      instruction,
    });
    const repairedChecks = groundingChecks(repaired.answer, verified, question);
    repair = { instruction, first_checks: checks, repaired_answer: repaired.answer, repaired_checks: repairedChecks };
    if (repaired.answer && !isRefusal(repaired.answer)) {
      finalAnswer = repaired.answer;
      checks = repairedChecks;
    }
  }

  if (!checks.passed) {
    // Do not ship an ungrounded paraphrase: fall back to the verified quotes themselves.
    return {
      ...base,
      ...llmTrace,
      repair,
      checks,
      mode: "EXTRACTIVE fallback (rule-based quotes; LLM answer failed grounding check)",
      answer: verified.map((v) => `"${v.quote.replace(/\s+/g, " ")}"`).join(" "),
      evidence: verified.map((v) => ({ quote: v.quote, chunk_id: v.source.id })),
      citations: uniqueCitations(verified),
    };
  }

  const supporting = supportingEvidence(finalAnswer, verified);
  return {
    ...base,
    ...llmTrace,
    repair,
    checks,
    mode: `LLM (local: ollama/${LLM_MODEL})${repair ? ", after 1 grounding repair" : ""}`,
    answer: finalAnswer,
    evidence: supporting.map((v) => ({ quote: v.quote, chunk_id: v.source.id })),
    citations: uniqueCitations(supporting),
  };
}

function pick(h) {
  return {
    id: h.id,
    document_title: h.document_title,
    heading: h.heading,
    source_file: h.source_file,
    page: h.page,
    score: h.score,
  };
}

function printResult(r, i) {
  console.log(`\n=== Q${i + 1}: ${r.question}`);
  console.log(`retrieved: ${r.retrieved.map((h) => `${h.id} (${h.score})`).join(", ")}`);
  console.log(`mode: ${r.mode}`);
  console.log(`answer: ${r.answer}`);
  for (const e of r.evidence) console.log(`  evidence (${e.chunk_id}): "${e.quote}"`);
  if (r.rejected_evidence?.length) console.log(`  rejected (not found in chunks): ${JSON.stringify(r.rejected_evidence)}`);
  if (r.repair) console.log(`  first LLM answer: ${r.llm_answer}\n  repair: ${r.repair.instruction}`);
  if (r.checks && !r.checks.passed) console.log(`  checks: ${JSON.stringify(r.checks)}`);
  for (const c of r.citations) console.log(`  [${c.ref}] ${citationOf(c)}`);
}

function writeReport(results) {
  const meta = {
    generated_at: new Date().toISOString(),
    pipeline:
      "deterministic: embed → retrieve → score gate → local LLM (JSON evidence + answer) → " +
      "code-verified evidence → code-built citations → grounding checks",
    embedding_model: `${LOCAL_EMBEDDING_MODEL} (local, 384d)`,
    vector_store: `Pinecone ${INDEX_NAME} / ${NAMESPACE}`,
    llm: `ollama/${LLM_MODEL} (local)`,
    llm_options: LLM_OPTIONS,
    top_k: TOP_K,
    min_score: MIN_SCORE,
    paid_apis_used: false,
  };
  writeFileSync(
    join(ROOT, `${REPORT_BASE}.json`),
    JSON.stringify({ ...meta, results }, null, 2) + "\n",
  );

  const md = [
    "# M4.3.2 — Deterministic RAG report (Smart Wardrobe)",
    "",
    `- Generated: ${meta.generated_at}`,
    `- Pipeline: ${meta.pipeline}`,
    `- Embeddings: ${meta.embedding_model}`,
    `- Vector store: ${meta.vector_store}`,
    `- LLM: ${meta.llm}, temperature 0, seed 42`,
    `- Retrieval: top-${TOP_K}, refusal gate at cosine < ${MIN_SCORE}`,
    "- Evidence: every quote is checked to occur verbatim in a retrieved chunk; citations are the chunks containing verified quotes (generated by code, not the model)",
    "- Paid APIs used: none",
    "",
  ];
  results.forEach((r, i) => {
    md.push(`## Q${i + 1}. ${r.question}`, "");
    md.push(`**Expected:** ${r.expect} · **Mode:** ${r.mode} · **Top score:** ${r.top_score}`, "");
    md.push(`**Answer:** ${r.answer}`, "");
    if (r.evidence.length) {
      md.push("**Verified evidence:**");
      for (const e of r.evidence) md.push(`> ${e.quote.replace(/\s+/g, " ")} — \`${e.chunk_id}\``);
      md.push("");
    }
    if (r.citations.length) {
      md.push("**Sources:**");
      for (const c of r.citations) md.push(`- [${c.ref}] ${citationOf(c)}`);
    } else {
      md.push("**Sources:** none — refused because the documents do not support an answer.");
    }
    if (r.checks) {
      md.push(
        "",
        `**Grounding checks:** ${r.checks.passed ? "passed" : "FAILED"}${r.repair ? " after 1 repair" : ""} (hedge dropped: ${r.checks.hedge_dropped}, ` +
          `unsupported terms: ${r.checks.unsupported_terms.length ? r.checks.unsupported_terms.join(", ") : "none"})`,
      );
    }
    if (r.repair) {
      md.push("", `<sub>First LLM answer (rejected): ${r.llm_answer} — repair: ${r.repair.instruction}</sub>`);
    }
    md.push("", `<sub>Retrieved: ${r.retrieved.map((h) => `${h.id} (${h.score})`).join(", ")}</sub>`, "");
  });
  writeFileSync(join(ROOT, `${REPORT_BASE}.md`), md.join("\n"));
}

async function main() {
  loadEnvLocal();
  const args = process.argv.slice(2);
  const retrieveOnly = args.includes("--retrieve-only");
  const adHoc = args.filter((a) => !a.startsWith("--")).join(" ").trim();
  const questions = adHoc ? [{ question: adHoc, expect: "n/a" }] : QUESTIONS;

  const ns = await getNamespace();

  if (retrieveOnly) {
    for (const { question, expect } of questions) {
      const hits = await retrieve(ns, question);
      console.log(`\n[${expect}] ${question}`);
      for (const h of hits) console.log(`  ${h.score.toFixed(4)}  ${h.id}  ${h.heading}`);
    }
    return;
  }

  await assertOllamaReady();
  const results = [];
  for (const [i, q] of questions.entries()) {
    const r = { expect: q.expect, ...(await answer(ns, q.question)) };
    printResult(r, i);
    results.push(r);
  }

  if (!adHoc) {
    writeReport(results);
    console.log(`\n✓ Report written to ${REPORT_BASE}.json and ${REPORT_BASE}.md`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
