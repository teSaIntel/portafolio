-- Production storage for the knowledge layer: one Postgres 16 with pgvector (documents) and Apache AGE (graph).
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;          -- needs shared_preload_libraries = 'age'
CREATE SCHEMA IF NOT EXISTS kb;
CREATE SCHEMA IF NOT EXISTS kg;

-- ---------------------------------------------------------------- knowledge base
CREATE TABLE kb.doc (
  id        bigserial PRIMARY KEY,
  key       text UNIQUE NOT NULL,            -- kind|bug|run|ts
  hash      text NOT NULL,                   -- content hash: re-ingest is idempotent
  kind      text NOT NULL CHECK (kind IN ('report','thread_comment','run_comment')),
  bug       text, run text, author text, ts timestamptz NOT NULL,
  domain    text, cycle text, config text, status text,
  quality   text NOT NULL CHECK (quality IN ('ok','short','noise')),
  body      text NOT NULL
);
CREATE TABLE kb.chunk (
  id        bigserial PRIMARY KEY,
  doc_id    bigint REFERENCES kb.doc(id) ON DELETE CASCADE,
  ord       int NOT NULL,
  text      text NOT NULL,                   -- <= 1100 chars, 120 overlap
  tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  vec       halfvec(1024)                    -- only for quality-ok docs; NULL until the incremental embed runs
);
CREATE INDEX chunk_tsv ON kb.chunk USING gin (tsv);
CREATE INDEX chunk_vec ON kb.chunk USING hnsw (vec halfvec_cosine_ops);
CREATE TABLE kb.entity (
  doc_id bigint REFERENCES kb.doc(id) ON DELETE CASCADE,
  kind   text NOT NULL,                      -- ticket | hex | path | symbol | logline
  value  text NOT NULL
);
CREATE INDEX entity_kv ON kb.entity (kind, value);

-- ---------------------------------------------------------------- curated graph (source of truth)
CREATE TABLE kg.node (
  id    text PRIMARY KEY,                    -- Label:key
  label text NOT NULL,
  props jsonb NOT NULL,
  hash  text NOT NULL                        -- node hash diff: new / changed / removed per build
);
CREATE TABLE kg.edge (
  src  text NOT NULL REFERENCES kg.node(id) ON DELETE CASCADE,
  rel  text NOT NULL,
  dst  text NOT NULL REFERENCES kg.node(id) ON DELETE CASCADE,
  prov text NOT NULL,                        -- tracker_link | enrich | test_registry | inferred_majority | text_mention
  conf real NOT NULL,
  PRIMARY KEY (src, rel, dst)
);
CREATE TABLE kg.run (                        -- one row per build, with the purify report (counts per rule)
  id bigserial PRIMARY KEY, at timestamptz DEFAULT now(), report jsonb NOT NULL
);
-- The openCypher projection (graph 'hgraph') is dropped and rebuilt from kg.* on every build (~10 s).
