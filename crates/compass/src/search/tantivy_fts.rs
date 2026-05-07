// search/tantivy_fts.rs — Full-text search + precomputed bitset faceting via Tantivy.
//
// Performance architecture:
//   - Full-text search: Tantivy's inverted index (BM25 scoring, sub-ms for any dataset size)
//   - Facet counting: PRECOMPUTED BITSETS. At index time, we build one bitset per unique
//     metadata value (e.g. one bitset for department="Legal"). At query time, we AND the
//     query's result bitset with each precomputed bitset and popcount.
//     This gives microsecond faceting even at millions of documents.
//
// How bitset faceting works:
//   At index time:
//     facetBitsets = { "department": { "Legal": BitSet([0,1,4,7,...]), "Eng": BitSet([2,3,...]) } }
//   At query time:
//     queryBits = search(query)         // bitset of matching doc IDs
//     legalCount = (queryBits AND facetBitsets["department"]["Legal"]).popcount()
//     // ^ This is ~20 microseconds for 250K documents

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

// ── Bitset implementation ────────────────────────────────────────────────────
// A compact bitset stored as a Vec<u64>. Each u64 holds 64 bits.
// This is the core data structure that makes faceting fast.

#[derive(Clone, Debug)]
pub struct BitSet {
    /// Each u64 stores 64 bits. words[0] covers bits 0-63, words[1] covers 64-127, etc.
    words: Vec<u64>,
    /// Total number of bits (= total number of documents in the collection)
    len: usize,
}

impl BitSet {
    /// Create a new bitset with all bits set to 0 (nothing matches).
    fn new(num_bits: usize) -> Self {
        // Ceiling division: how many u64 words we need to cover all bits
        let num_words = (num_bits + 63) / 64;
        Self {
            words: vec![0u64; num_words],
            len: num_bits,
        }
    }

    /// Create a bitset with ALL bits set to 1 (everything matches).
    /// Used for unfiltered facet queries where every document counts.
    fn all(num_bits: usize) -> Self {
        let num_words = (num_bits + 63) / 64;
        let mut words = vec![u64::MAX; num_words];
        // Clear the extra trailing bits in the last word so popcount stays accurate
        let trailing = num_bits % 64;
        if trailing > 0 && !words.is_empty() {
            let last = words.len() - 1;
            words[last] = (1u64 << trailing) - 1;
        }
        Self {
            words,
            len: num_bits,
        }
    }

    /// Set a single bit to 1 (mark document at this position as matching).
    #[inline]
    fn set(&mut self, bit: usize) {
        if bit < self.len {
            // bit >> 6 = which u64 word (dividing by 64)
            // bit & 63 = which bit within that word (modulo 64)
            self.words[bit >> 6] |= 1u64 << (bit & 63);
        }
    }

    /// AND two bitsets together, producing a new bitset.
    /// This is the hot path — called once per facet value per query.
    /// Each iteration processes 64 documents in a single CPU instruction.
    #[inline]
    fn and(&self, other: &BitSet) -> BitSet {
        let min_len = self.words.len().min(other.words.len());
        let mut result = Vec::with_capacity(min_len);
        for i in 0..min_len {
            // Compiles down to a single AND instruction per 64 documents
            result.push(self.words[i] & other.words[i]);
        }
        BitSet {
            words: result,
            len: self.len.min(other.len),
        }
    }

    /// Count the number of set bits (1s) in the entire bitset.
    /// Uses the CPU's native POPCNT instruction for maximum speed.
    #[inline]
    fn popcount(&self) -> u64 {
        // count_ones() compiles to hardware POPCNT — processes 64 bits per clock cycle
        self.words.iter().map(|w| w.count_ones() as u64).sum()
    }
}

// ── Precomputed facet bitsets ────────────────────────────────────────────────
// Built once at index time, reused for every facet query.
// Structure: { "department" => { "Legal" => BitSet, "Eng" => BitSet }, ... }

#[derive(Clone, Debug)]
pub struct FacetBitsets {
    /// Nested map: field_name -> { value -> bitset of matching doc positions }
    pub groups: HashMap<String, HashMap<String, BitSet>>,
    /// Total number of documents (needed to create "all" bitsets for unfiltered queries)
    pub total_docs: usize,
}

// ── FtsState ─────────────────────────────────────────────────────────────────
// Holds everything needed to run full-text search and facet queries on a collection.

#[derive(Clone)]
pub struct FtsState {
    pub index: Index,
    /// Cached reader — created once, reused for all queries (avoids ~1ms overhead per query)
    pub reader: tantivy::IndexReader,
    // Field handles for the Tantivy schema
    pub id_field: Field,
    pub collection_field: Field,
    pub file_id_field: Field,
    pub chunk_index_field: Field,
    pub page_field: Field,
    pub text_field: Field,
    /// We store arbitrary metadata as a JSON string field (indexed per-key via facet bitsets)
    pub metadata_field: Field,
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
    existing_count: u64,
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

    // ── Precompute facet bitsets ──────────────────────────────────────────────
    // We need ALL chunks in the collection (existing + new) to build accurate bitsets.
    // For now, we rebuild bitsets from the chunks we have. On reload from disk,
    // the collection manager will call rebuild_facets() with all chunks.
    let total_docs = (existing_count as usize) + chunks.len();
    let facet_bitsets = build_facet_bitsets(chunks, existing_count as usize, total_docs);

    // Create a reader once, reuse for all queries
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;

    Ok(FtsState {
        index,
        reader,
        id_field: fields.id,
        collection_field: fields.collection,
        file_id_field: fields.file_id,
        chunk_index_field: fields.chunk_index,
        page_field: fields.page,
        text_field: fields.text,
        metadata_field: fields.metadata,
        facet_bitsets,
    })
}

/// Open an existing Tantivy index from disk (used on server restart).
pub fn open_index(dir: &Path) -> Result<FtsState, Box<dyn std::error::Error + Send + Sync>> {
    let index = Index::open_in_dir(dir)?;
    register_tokenizers(&index);

    let schema = index.schema();
    let id_field = schema.get_field("id")?;
    let collection_field = schema.get_field("collection")?;
    let file_id_field = schema.get_field("file_id")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let page_field = schema.get_field("page")?;
    let text_field = schema.get_field("text")?;
    let metadata_field = schema.get_field("metadata")?;

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;

    // Start with empty facet bitsets — caller must call rebuild_facets() after loading all chunks
    let facet_bitsets = FacetBitsets {
        groups: HashMap::new(),
        total_docs: 0,
    };

    Ok(FtsState {
        index,
        reader,
        id_field,
        collection_field,
        file_id_field,
        chunk_index_field,
        page_field,
        text_field,
        metadata_field,
        facet_bitsets,
    })
}

/// Build facet bitsets from a batch of chunks.
/// `offset` is the starting bit position (for appending to existing indices).
fn build_facet_bitsets(chunks: &[DocumentChunk], offset: usize, total_docs: usize) -> FacetBitsets {
    let mut groups: HashMap<String, HashMap<String, BitSet>> = HashMap::new();

    // Scan all chunks and set bits for each metadata key-value pair.
    // MetadataValue is converted to a string for facet grouping (e.g. Float(9.5) -> "9.5").
    for (i, chunk) in chunks.iter().enumerate() {
        let bit_pos = offset + i;
        for (key, value) in &chunk.metadata {
            let value_str = metadata_to_facet_string(value);
            groups
                .entry(key.clone())
                .or_default()
                .entry(value_str)
                .or_insert_with(|| BitSet::new(total_docs))
                .set(bit_pos);
        }
    }

    FacetBitsets { groups, total_docs }
}

/// Convert a MetadataValue to a string for facet grouping.
fn metadata_to_facet_string(val: &MetadataValue) -> String {
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
/// Metadata filtering is handled post-search by the collection manager (using the scoring
/// pipeline), so this function only does text-based search.
pub fn search(
    state: &FtsState,
    query_str: &str,
    filters: &HashMap<String, String>,
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

    // If there are metadata filters, combine them with the text query using BooleanQuery
    let query: Box<dyn tantivy::query::Query> = if filters.is_empty() {
        text_query
    } else {
        // Each filter becomes a MUST clause — all must match
        let mut clauses: Vec<(tantivy::query::Occur, Box<dyn tantivy::query::Query>)> = Vec::new();
        clauses.push((tantivy::query::Occur::Must, text_query));

        // Metadata filters are matched against stored text fields.
        // Since metadata is stored as JSON, we can't filter directly in Tantivy.
        // Instead, we apply metadata filtering post-search using the bitsets.
        // For now, we include the text query only and let the caller handle filtering.
        // TODO: implement metadata filtering via bitset post-filtering

        Box::new(tantivy::query::BooleanQuery::new(clauses))
    };

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
) -> Result<(HashMap<String, HashMap<String, u64>>, u64), Box<dyn std::error::Error + Send + Sync>>
{
    let start = std::time::Instant::now();
    let bs = &state.facet_bitsets;

    // For unfiltered queries, every document matches — use "all ones" bitset
    let query_bitset = if query_str.is_empty() || query_str == "*" {
        None // fast path: skip query execution entirely
    } else {
        // Execute the text query and build a bitset from matching doc IDs
        let searcher = state.reader.searcher();
        let query_parser = QueryParser::for_index(&state.index, vec![state.text_field]);

        let query: Box<dyn tantivy::query::Query> = match query_parser.parse_query(query_str) {
            Ok(q) => q,
            Err(_) => Box::new(tantivy::query::AllQuery),
        };

        let top_docs = searcher.search(&query, &TopDocs::with_limit(bs.total_docs))?;

        let mut result_bits = BitSet::new(bs.total_docs);
        for (_score, doc_address) in &top_docs {
            let doc: tantivy::TantivyDocument = searcher.doc(*doc_address)?;
            if let Some(tantivy::schema::OwnedValue::U64(id)) = doc.get_first(state.id_field) {
                result_bits.set(*id as usize);
            }
        }
        Some(result_bits)
    };

    // THE HOT PATH: bitset AND + popcount for each facet value
    let mut facets: HashMap<String, HashMap<String, u64>> = HashMap::new();

    for (group_name, value_bitsets) in &bs.groups {
        // If specific fields were requested, skip fields not in the list
        if !requested_fields.is_empty() && !requested_fields.contains(group_name) {
            continue;
        }

        let mut counts: HashMap<String, u64> = HashMap::new();
        for (value, value_bits) in value_bitsets {
            let count = match &query_bitset {
                // Unfiltered: just popcount the precomputed bitset directly
                None => value_bits.popcount(),
                // Filtered: AND with query results, then popcount the intersection
                Some(qb) => qb.and(value_bits).popcount(),
            };
            if count > 0 {
                counts.insert(value.clone(), count);
            }
        }
        facets.insert(group_name.clone(), counts);
    }

    let took_us = start.elapsed().as_micros() as u64;
    Ok((facets, took_us))
}
