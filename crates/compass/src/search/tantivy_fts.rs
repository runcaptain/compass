// search/tantivy_fts.rs — Full-text search + precomputed facet treemaps via Tantivy.
//
// Performance architecture:
//   - Full-text search: Tantivy's inverted index (BM25 scoring, sub-ms for any dataset size)
//   - Facet counting: precomputed roaring treemaps keyed by CHUNK ID, one per unique
//     metadata value. At query time each value's treemap is intersected with the
//     live-id universe (and the query's hit set, if any) and popcounted —
//     microsecond faceting independent of collection size, correct under
//     sparse/block-allocated ids and deletes.

use crate::models::{DocumentChunk, MetadataValue};
use std::collections::HashMap;
use std::path::Path;
use tantivy::collector::{Count, TopDocs};
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::tokenizer::{
    Language, LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer, TextAnalyzer,
};
use tantivy::{Index, IndexWriter, ReloadPolicy};

// ── Precomputed facet bitsets ────────────────────────────────────────────────
// Built once at index time, reused for every facet query.
// Structure: { "department" => { "Legal" => RoaringTreemap(chunk ids), ... } }

#[derive(Clone, Debug, Default)]
pub struct FacetBitsets {
    /// Nested map: field_name -> { value -> treemap of matching CHUNK IDS }.
    /// Keyed by chunk id (not doc position): ids are u64 and non-dense once
    /// block-allocated, and the query side has always intersected on the
    /// stored id — position-keyed dense bitsets silently misaligned.
    pub groups: HashMap<String, HashMap<String, roaring::RoaringTreemap>>,
}

impl FacetBitsets {
    /// Union another (older) facet map into this one. Appending a batch used
    /// to REPLACE the facet state with new-batch-only bitsets — facets went
    /// wrong after the second ingest batch, latent since v0.2.
    pub fn absorb(&mut self, older: &FacetBitsets) {
        for (field, vals) in &older.groups {
            let dst = self.groups.entry(field.clone()).or_default();
            for (value, tm) in vals {
                *dst.entry(value.clone()).or_default() |= tm;
            }
        }
    }

    /// Record one chunk's facet values (used by streaming rebuilds at load).
    pub fn insert_chunk(&mut self, chunk: &DocumentChunk) {
        insert_facets_for(&mut self.groups, chunk);
    }
}

fn insert_facets_for(
    groups: &mut HashMap<String, HashMap<String, roaring::RoaringTreemap>>,
    chunk: &DocumentChunk,
) {
    for (field, value) in &chunk.metadata {
        let repr = metadata_value_repr(value);
        groups
            .entry(field.clone())
            .or_default()
            .entry(repr)
            .or_default()
            .insert(chunk.id);
    }
}

// ── FtsState ─────────────────────────────────────────────────────────────────
// Holds everything needed to run full-text search and facet queries on a collection.

#[derive(Clone)]
pub struct FtsState {
    pub index: Index,
    /// Cached reader — created once, reused for all queries (avoids ~1ms overhead per query)
    pub reader: tantivy::IndexReader,
    // Field handles the query paths read. (The schema defines more columns —
    // collection/file_id/chunk_index/page/metadata — written at index time via
    // FtsFields; only these two are read back.)
    pub id_field: Field,
    pub text_field: Field,
    /// Precomputed bitsets for microsecond faceting
    pub facet_bitsets: FacetBitsets,
}

/// Internal struct to pass field handles out of schema creation
struct FtsFields {
    id: Field,
    collection: Field,
    file_id: Field,
    chunk_index: Field,
    page: Field,
    text: Field,
    metadata: Field,
}

/// Build the Tantivy schema for a Compass collection.
/// Uses English stemming on the text field for better recall (e.g. "running" matches "run").
fn build_schema() -> (Schema, FtsFields) {
    let mut builder = Schema::builder();

    // Configure English stemming tokenizer for the main text field
    let stemmed_indexing = TextFieldIndexing::default()
        .set_tokenizer("en_stem")
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);

    let stemmed_text = TextOptions::default()
        .set_stored()
        .set_indexing_options(stemmed_indexing);

    let id = builder.add_u64_field("id", STORED | INDEXED);
    let collection = builder.add_text_field("collection", STRING | STORED);
    let file_id = builder.add_text_field("file_id", STRING | STORED);
    let chunk_index = builder.add_u64_field("chunk_index", STORED | INDEXED);
    let page = builder.add_u64_field("page", STORED | INDEXED);
    let text = builder.add_text_field("text", stemmed_text);
    // Metadata stored as a JSON blob for retrieval; faceting uses the bitsets, not this field
    let metadata = builder.add_text_field("metadata", STORED);

    let schema = builder.build();
    let fields = FtsFields {
        id,
        collection,
        file_id,
        chunk_index,
        page,
        text,
        metadata,
    };
    (schema, fields)
}

/// Register the English stemmer tokenizer on a Tantivy index.
/// Must be called before writing or reading, and must match the tokenizer name in the schema.
fn register_tokenizers(index: &Index) {
    let en_stem = TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(RemoveLongFilter::limit(40))
        .filter(LowerCaser)
        .filter(Stemmer::new(Language::English))
        .build();
    index.tokenizers().register("en_stem", en_stem);
}

/// Build a disk-backed Tantivy index from a batch of chunks.
/// Also precomputes the facet bitsets for all metadata keys.
///
/// `dir` is the directory where Tantivy will write its index files.
/// If the directory already exists and has an index, this appends to it.
pub fn build_index(
    dir: &Path,
    chunks: &[DocumentChunk],
) -> Result<FtsState, Box<dyn std::error::Error + Send + Sync>> {
    let (schema, fields) = build_schema();

    // Create (or open) a disk-backed index in the given directory
    let index = if dir.join("meta.json").exists() {
        // Index already exists on disk — open it and append
        let index = Index::open_in_dir(dir)?;
        register_tokenizers(&index);
        index
    } else {
        // Fresh index — create the directory and initialize
        std::fs::create_dir_all(dir)?;
        let index = Index::create_in_dir(dir, schema)?;
        register_tokenizers(&index);
        index
    };

    // Multi-threaded writer for faster bulk indexing
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get().min(4))
        .unwrap_or(2);
    let mut writer: IndexWriter = index.writer_with_num_threads(num_threads, 256_000_000)?;

    // Add each chunk as a Tantivy document
    for chunk in chunks {
        let mut doc = tantivy::TantivyDocument::default();
        doc.add_u64(fields.id, chunk.id);
        doc.add_text(fields.collection, &chunk.collection);
        doc.add_text(fields.file_id, &chunk.file_id);
        doc.add_u64(fields.chunk_index, chunk.chunk_index as u64);
        if let Some(page) = chunk.page {
            doc.add_u64(fields.page, page as u64);
        }
        doc.add_text(fields.text, &chunk.text);
        // Store metadata as JSON for retrieval
        let meta_json = serde_json::to_string(&chunk.metadata).unwrap_or_default();
        doc.add_text(fields.metadata, &meta_json);
        writer.add_document(doc)?;
    }

    writer.commit()?;

    // Facet treemaps for THIS batch only. Callers accumulate: ingest absorbs
    // the prior state; the load/rebuild scans reconstruct from all live chunks.
    let facet_bitsets = build_facet_bitsets(chunks);

    // Create a reader once, reuse for all queries
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;

    Ok(FtsState {
        index,
        reader,
        id_field: fields.id,
        text_field: fields.text,
        facet_bitsets,
    })
}

/// Open an existing Tantivy index from disk (used on server restart).
pub fn open_index(dir: &Path) -> Result<FtsState, Box<dyn std::error::Error + Send + Sync>> {
    let index = Index::open_in_dir(dir)?;
    register_tokenizers(&index);

    let schema = index.schema();
    let id_field = schema.get_field("id")?;
    let text_field = schema.get_field("text")?;

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;

    // Start with empty facets — the collection manager rebuilds them in its
    // load-time chunk scan (streaming, chunk-id keyed).
    let facet_bitsets = FacetBitsets::default();

    Ok(FtsState {
        index,
        reader,
        id_field,
        text_field,
        facet_bitsets,
    })
}

/// Build facet bitsets from a batch of chunks.
/// `offset` is the starting bit position (for appending to existing indices).
fn build_facet_bitsets(chunks: &[DocumentChunk]) -> FacetBitsets {
    let mut fb = FacetBitsets::default();
    for chunk in chunks {
        fb.insert_chunk(chunk);
    }
    fb
}

/// Convert a MetadataValue to a string for facet grouping.
fn metadata_value_repr(val: &MetadataValue) -> String {
    match val {
        MetadataValue::String(s) => s.clone(),
        MetadataValue::Int(i) => i.to_string(),
        MetadataValue::Float(f) => f.to_string(),
        MetadataValue::Bool(b) => b.to_string(),
        MetadataValue::StringList(list) => list.join(","),
    }
}

/// Run a full-text search query. Returns (matching doc IDs + scores, total count, microseconds).
///
/// Metadata filtering is handled by the collection manager (roaring filter-index
/// pushdown + scoring pipeline), so this function only does text-based search.
pub fn search(
    state: &FtsState,
    query_str: &str,
    limit: usize,
) -> Result<(Vec<(u64, f32)>, usize, u64), Box<dyn std::error::Error + Send + Sync>> {
    let start = std::time::Instant::now();
    let searcher = state.reader.searcher();

    // Build the text query (searches the "text" field by default)
    let query_parser = QueryParser::for_index(&state.index, vec![state.text_field]);

    let text_query: Box<dyn tantivy::query::Query> = if query_str.is_empty() || query_str == "*" {
        // Empty query = match everything
        Box::new(tantivy::query::AllQuery)
    } else {
        match query_parser.parse_query(query_str) {
            Ok(q) => q,
            Err(_) => {
                // If query parsing fails, fall back to a simple term query
                let term = tantivy::Term::from_field_text(state.text_field, query_str);
                Box::new(tantivy::query::TermQuery::new(
                    term,
                    IndexRecordOption::Basic,
                ))
            }
        }
    };

    let query: Box<dyn tantivy::query::Query> = text_query;

    // Execute search: get top results + total count in a single pass
    let (top_docs, total_count) = searcher.search(&query, &(TopDocs::with_limit(limit), Count))?;

    // Extract document IDs and scores from results
    let mut results: Vec<(u64, f32)> = Vec::with_capacity(top_docs.len());
    for (score, doc_address) in top_docs {
        let doc: tantivy::TantivyDocument = searcher.doc(doc_address)?;
        if let Some(id) = doc.get_first(state.id_field).and_then(|v| match v {
            tantivy::schema::OwnedValue::U64(n) => Some(*n),
            _ => None,
        }) {
            results.push((id, score));
        }
    }

    let took_us = start.elapsed().as_micros() as u64;
    Ok((results, total_count, took_us))
}

/// Compute facet counts using precomputed bitsets.
///
/// For unfiltered queries (empty or "*"), we just popcount each precomputed bitset.
/// For filtered queries, we build a query result bitset, AND it with each facet bitset,
/// and popcount the intersection.
pub fn get_facets(
    state: &FtsState,
    query_str: &str,
    requested_fields: &[String],
    live: &roaring::RoaringTreemap,
) -> Result<(HashMap<String, HashMap<String, u64>>, u64), Box<dyn std::error::Error + Send + Sync>>
{
    let start = std::time::Instant::now();
    let bs = &state.facet_bitsets;

    // Text-filtered queries build a treemap of matching CHUNK IDS; unfiltered
    // queries skip query execution entirely. Counts always intersect with the
    // LIVE universe, so soft-deleted chunks never inflate facets.
    let query_ids: Option<roaring::RoaringTreemap> = if query_str.is_empty() || query_str == "*" {
        None
    } else {
        let searcher = state.reader.searcher();
        let query_parser = QueryParser::for_index(&state.index, vec![state.text_field]);
        let query: Box<dyn tantivy::query::Query> = match query_parser.parse_query(query_str) {
            Ok(q) => q,
            Err(_) => Box::new(tantivy::query::AllQuery),
        };
        let top_docs = searcher.search(&query, &TopDocs::with_limit(usize::MAX >> 32))?;
        let mut ids = roaring::RoaringTreemap::new();
        for (_score, doc_address) in &top_docs {
            let doc: tantivy::TantivyDocument = searcher.doc(*doc_address)?;
            if let Some(tantivy::schema::OwnedValue::U64(id)) = doc.get_first(state.id_field) {
                ids.insert(*id);
            }
        }
        Some(ids)
    };

    let mut out: HashMap<String, HashMap<String, u64>> = HashMap::new();
    for (field, values) in &bs.groups {
        if !requested_fields.is_empty() && !requested_fields.contains(field) {
            continue;
        }
        let mut counts: HashMap<String, u64> = HashMap::new();
        for (value, tm) in values {
            let mut hit = tm & live;
            if let Some(q) = &query_ids {
                hit &= q;
            }
            let n = hit.len();
            if n > 0 {
                counts.insert(value.clone(), n);
            }
        }
        if !counts.is_empty() {
            out.insert(field.clone(), counts);
        }
    }
    Ok((out, start.elapsed().as_micros() as u64))
}
