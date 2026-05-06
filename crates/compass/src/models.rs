// models.rs — Core data types for Compass v2.
//
// Adds: typed metadata, named vector spaces, document relationships,
// scoring config (recency, boost, relationship), TAMS-compatible hierarchy.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── MetadataValue ─────────────────────────────────────────────────────────
// Typed metadata values. Uses serde's untagged enum so JSON stays natural:
//   "department": "Legal"          → String
//   "score": 9.5                   → Float
//   "priority": 3                  → Int
//   "active": true                 → Bool
//   "created_at": "2026-01-01..." → String (parsed as Timestamp at query time)
//   "tags": ["sports", "goals"]   → StringList

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MetadataValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    StringList(Vec<String>),
}

impl MetadataValue {
    /// Try to extract a float value (works for Int, Float, and numeric strings)
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetadataValue::Float(f) => Some(*f),
            MetadataValue::Int(i) => Some(*i as f64),
            MetadataValue::String(s) => s.parse::<f64>().ok(),
            _ => None,
        }
    }

    /// Try to extract a string value
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

// ── VectorSpaceConfig ─────────────────────────────────────────────────────
// Each collection can have multiple named vector spaces, each backed by its
// own USearch HNSW index. This lets you run multiple embedding models on the
// same collection (e.g. BGE-small for text, CLIP for images) and swap models
// without re-indexing everything at once.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSpaceConfig {
    /// Dimensionality of vectors in this space (e.g. 384 for BGE-small)
    pub dims: usize,
    /// Which embedding model produced these vectors
    pub model: String,
    /// "active" = ready for queries, "building" = rebuild in progress, "deprecated" = old
    #[serde(default = "default_space_status")]
    pub status: String,
}

fn default_space_status() -> String {
    "active".to_string()
}

// ── DocumentChunk ─────────────────────────────────────────────────────────
// A single indexed chunk. Now with typed metadata, relationship fields,
// and support for named vector spaces (multiple embeddings per chunk).

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    /// Unique numeric ID assigned at ingest time (auto-incremented per collection)
    pub id: u64,
    /// Which collection this chunk belongs to
    pub collection: String,
    /// Source file identifier (e.g. S3 key, Google Drive file ID)
    pub file_id: String,
    /// Position of this chunk within the source file (0 = first chunk)
    pub chunk_index: u32,
    /// Page number for paginated documents (PDF, DOCX)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<u32>,
    /// The text content (FTS-indexed)
    pub text: String,
    /// Typed key-value metadata. Each unique key becomes a facetable field.
    #[serde(default)]
    pub metadata: HashMap<String, MetadataValue>,
    /// Document type for hierarchy: "source", "flow", "segment", or "chunk" (default)
    #[serde(default = "default_doc_type")]
    pub doc_type: String,
    /// Parent chunk ID (for hierarchical docs, e.g. TAMS Source → Flow → Segment)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<u64>,
    /// Groups siblings at the same hierarchy level (chunks sharing a parent share a group_id)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_id: Option<String>,
    /// Named embeddings: {"bge-small": [0.1, ...], "clip-vision": [0.2, ...]}
    /// If empty/None, Compass computes using the default vector space's model.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub embeddings: HashMap<String, Vec<f32>>,
    /// Legacy single embedding field (backward compat with v1 API).
    /// Mapped to the default vector space at ingest time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

fn default_doc_type() -> String {
    "chunk".to_string()
}

// ── Collection ────────────────────────────────────────────────────────────
// Now includes named vector spaces instead of a single embedding_dims.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    pub name: String,
    pub created_at: DateTime<Utc>,
    /// Named vector spaces: {"bge-small": {dims: 384, model: "...", status: "active"}, ...}
    #[serde(default)]
    pub vector_spaces: HashMap<String, VectorSpaceConfig>,
    /// Which vector space to use when the caller doesn't specify one
    #[serde(default)]
    pub default_vector_space: Option<String>,
    /// Legacy field for backward compat (maps to the default space's dims)
    #[serde(default = "default_dims")]
    pub embedding_dims: usize,
    pub chunk_count: u64,
    #[serde(default)]
    pub config: CollectionConfig,
}

fn default_dims() -> usize {
    384
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CollectionConfig {
    #[serde(default = "default_embed_model")]
    pub embed_model: String,
}

fn default_embed_model() -> String {
    "bge-small".to_string()
}

// ── API Request Types ─────────────────────────────────────────────────────

/// POST /collections
#[derive(Debug, Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    /// Initial vector spaces. If omitted, creates one "default" space with 384 dims.
    pub vector_spaces: Option<HashMap<String, VectorSpaceConfig>>,
    /// Legacy: embedding_dims (creates a "default" space with these dims)
    pub embedding_dims: Option<usize>,
    pub config: Option<CollectionConfig>,
}

/// POST /collections/:name/ingest
#[derive(Debug, Deserialize)]
pub struct IngestRequest {
    pub chunks: Vec<IngestChunk>,
}

/// A single chunk in an ingest request (before Compass assigns an ID)
#[derive(Debug, Deserialize)]
pub struct IngestChunk {
    /// Caller-assigned ID for cross-referencing within the same batch
    #[serde(default)]
    pub client_id: Option<String>,
    pub file_id: String,
    pub chunk_index: u32,
    #[serde(default)]
    pub page: Option<u32>,
    pub text: String,
    #[serde(default)]
    pub metadata: HashMap<String, MetadataValue>,
    /// Document type: "source", "flow", "segment", or "chunk" (default)
    #[serde(default = "default_doc_type")]
    pub doc_type: String,
    /// Parent chunk ID from a PREVIOUS ingest (Compass-assigned u64)
    #[serde(default)]
    pub parent_id: Option<u64>,
    /// Parent's client_id within THIS batch (resolved by Compass at write time)
    #[serde(default)]
    pub parent_ref: Option<String>,
    /// Group ID for sibling grouping (chunks sharing a parent share a group_id)
    #[serde(default)]
    pub group_id: Option<String>,
    /// Named embeddings: {"bge-small": [0.1, ...], "clip-vision": [0.2, ...]}
    #[serde(default)]
    pub embeddings: HashMap<String, Vec<f32>>,
    /// Legacy single embedding (mapped to the default vector space)
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

/// POST /collections/:name/search
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    /// "fts", "semantic", or "hybrid" (default: "hybrid")
    #[serde(default = "default_search_mode")]
    pub mode: String,
    /// Which named vector space to query (defaults to collection's default_vector_space)
    #[serde(default)]
    pub vector_space: Option<String>,
    /// Number of results to return (default: 10)
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Pre-computed query vector. When set, semantic search uses this directly
    /// and skips in-process embedding. Useful for benchmarks and cases where
    /// the caller has already embedded the query.
    #[serde(default, alias = "vector")]
    pub query_vector: Option<Vec<f32>>,
    /// Metadata filters (exact match, range, contains, set membership).
    /// Backward compatible: plain values are exact match.
    #[serde(default)]
    pub filters: HashMap<String, FilterValue>,
    /// Weights for blending FTS vs semantic scores in hybrid mode.
    #[serde(default)]
    pub score_weights: Option<ScoreWeights>,
    /// Recency decay configuration (full control)
    #[serde(default)]
    pub recency: Option<RecencyConfig>,
    /// Recency preset: "recent", "mild", "aggressive", or "archive".
    /// Shorthand that expands to a RecencyConfig. Requires `recency_field`.
    /// Ignored if `recency` is also set (explicit config wins).
    #[serde(default)]
    pub recency_preset: Option<String>,
    /// Which metadata timestamp field to use with recency_preset (e.g. "created_at")
    #[serde(default)]
    pub recency_field: Option<String>,
    /// Metadata field boost factors
    #[serde(default)]
    pub boosts: Vec<BoostConfig>,
    /// Relationship-based score boosting
    #[serde(default)]
    pub relationship_boost: Option<RelationshipBoostConfig>,
}

fn default_search_mode() -> String {
    "hybrid".to_string()
}

fn default_top_k() -> usize {
    10
}

/// Recency decay: newer documents score higher.
/// Formula: decay = max(min_score, 2^(-age_days / half_life_days))
#[derive(Debug, Clone, Deserialize)]
pub struct RecencyConfig {
    /// Which metadata timestamp field to use (e.g. "created_at")
    pub field: String,
    /// Score halves every N days (e.g. 30 = a 30-day-old doc scores 0.5x)
    pub half_life_days: f64,
    /// Floor value so old docs never go to zero (default: 0.1)
    #[serde(default = "default_min_score")]
    pub min_score: f64,
}

impl RecencyConfig {
    /// Expand a preset name into a full RecencyConfig.
    ///   "recent"     — 7-day half-life, floor 0.2 (news, feeds, tickets)
    ///   "mild"       — 30-day half-life, floor 0.3 (docs, reports)
    ///   "aggressive" — 3-day half-life, floor 0.05 (real-time, alerts)
    ///   "archive"    — 90-day half-life, floor 0.5 (long-lived content)
    pub fn from_preset(name: &str, field: String) -> Option<Self> {
        let (half_life_days, min_score) = match name {
            "recent" => (7.0, 0.2),
            "mild" => (30.0, 0.3),
            "aggressive" => (3.0, 0.05),
            "archive" => (90.0, 0.5),
            _ => return None,
        };
        Some(Self { field, half_life_days, min_score })
    }
}

fn default_min_score() -> f64 {
    0.1
}

/// Metadata boost: multiply the score when a field matches a condition.
#[derive(Debug, Clone, Deserialize)]
pub struct BoostConfig {
    /// Metadata field name to check
    pub field: String,
    /// Exact string match (for string fields)
    #[serde(default)]
    pub value: Option<String>,
    /// Greater-than-or-equal (for numeric fields)
    #[serde(default)]
    pub gte: Option<f64>,
    /// Less-than-or-equal (for numeric fields)
    #[serde(default)]
    pub lte: Option<f64>,
    /// Multiplicative boost weight (e.g. 2.0 = double the score)
    #[serde(default = "default_boost_weight")]
    pub weight: f64,
}

fn default_boost_weight() -> f64 {
    1.0
}

/// Relationship boost: boost scores based on parent/sibling matches.
/// Formula: factor = 1.0 + parent_weight * I(parent_in_results) + sibling_weight * agg(sibling_scores)
#[derive(Debug, Clone, Deserialize)]
pub struct RelationshipBoostConfig {
    /// Boost factor when the parent also appears in results (default: 0.3)
    #[serde(default = "default_parent_weight")]
    pub parent_weight: f64,
    /// Boost factor for sibling score aggregation (default: 0.1)
    #[serde(default = "default_sibling_weight")]
    pub sibling_weight: f64,
    /// How to aggregate sibling scores: "max", "avg", or "sum" (default: "max")
    #[serde(default = "default_agg_mode")]
    pub mode: String,
}

fn default_parent_weight() -> f64 {
    0.3
}

fn default_sibling_weight() -> f64 {
    0.1
}

fn default_agg_mode() -> String {
    "max".to_string()
}

// ── Filter Types ─────────────────────────────────────────────────────────
// Rich metadata filters: range (gte/lte), array contains, set membership.
// Backward compatible: a plain MetadataValue in JSON still works as exact match.

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FilterCondition {
    #[serde(default)]
    pub gte: Option<f64>,
    #[serde(default)]
    pub lte: Option<f64>,
    #[serde(default)]
    pub contains: Option<String>,
    #[serde(default, rename = "in")]
    pub in_values: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum FilterValue {
    Condition(FilterCondition),
    Exact(MetadataValue),
}

// ── Score Weights ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct ScoreWeights {
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f64,
    #[serde(default = "default_score_weight")]
    pub fts_weight: f64,
    #[serde(default = "default_score_weight")]
    pub semantic_weight: f64,
}

fn default_rrf_k() -> f64 {
    60.0
}

fn default_score_weight() -> f64 {
    1.0
}

/// GET /collections/:name/facets
#[derive(Debug, Deserialize)]
pub struct FacetRequest {
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub fields: Vec<String>,
}

/// POST /collections/:name/vector-spaces
#[derive(Debug, Deserialize)]
pub struct AddVectorSpaceRequest {
    pub name: String,
    pub dims: usize,
    pub model: String,
}

/// POST /collections/:name/vector-spaces/:space/rebuild
#[derive(Debug, Deserialize)]
pub struct RebuildRequest {
    /// External embedding endpoint URL. If omitted, uses built-in Candle embedder.
    #[serde(default)]
    pub embed_endpoint: Option<String>,
    /// Batch size for external endpoint calls (default: 64)
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

fn default_batch_size() -> usize {
    64
}

/// PUT /collections/:name/default-vector-space
#[derive(Debug, Deserialize)]
pub struct SetDefaultSpaceRequest {
    pub name: String,
}

// ── API Response Types ────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchHit>,
    pub total: usize,
    pub took_us: u64,
    pub mode: String,
}

#[derive(Debug, Serialize)]
pub struct SearchHit {
    pub chunk: DocumentChunk,
    /// Final score after all scoring pipeline stages
    pub score: f32,
    /// Which search method found this: "fts", "semantic", or "both"
    pub source: String,
}

#[derive(Debug, Serialize)]
pub struct FacetResponse {
    pub facets: HashMap<String, HashMap<String, u64>>,
    pub took_us: u64,
}

#[derive(Debug, Serialize)]
pub struct IngestResponse {
    pub indexed: usize,
    /// Map of client_id -> Compass-assigned chunk ID (for parent referencing in future ingests)
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub id_map: HashMap<String, u64>,
    pub took_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct CollectionInfo {
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub embedding_dims: usize,
    pub chunk_count: u64,
    pub vector_spaces: HashMap<String, VectorSpaceConfig>,
    pub default_vector_space: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct VectorSpaceInfo {
    pub name: String,
    pub dims: usize,
    pub model: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RebuildStatus {
    pub status: String,
    pub embedded: u64,
    pub total: u64,
    pub percent: f64,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub collections: usize,
    pub version: String,
}
