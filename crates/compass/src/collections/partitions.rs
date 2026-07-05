// collections/partitions.rs — tenant-partitioned collections (roadmap Phase 6).
//
// Design: a partition IS a full internal collection. Each partition gets its
// own namespace — LSM manifest, WAL, segments, Tantivy dir, vector files,
// attach/evict lifecycle — by reusing the existing per-collection engine
// wholesale. Phase 6 is a ROUTER at the manager entry points, not a new
// engine:
//
//   - `create` with `config.partition_by = "tenant_id"` marks the parent.
//     The parent namespace holds config + the shared id allocator; chunk data
//     lives only in partitions.
//   - Ingest groups chunks by `metadata[partition_by]` and delegates each
//     group to the partition's namespace, auto-creating it on first sight.
//   - Search/deletes require a filter on the partition field and route to
//     exactly the named partitions (set membership fans out, capped).
//   - Chunk ids are COLLECTION-unique: every partition claims id blocks from
//     the PARENT's allocator (in local mode too — the allocator is just a
//     CAS-updated file; this does not create WAL/manifest objects).
//
// Why this shape: per-tenant cost isolation falls out of the existing
// machinery. A query for tenant T attaches T's partition only; LRU eviction
// and refresh scale with the HOT tenant set, not the collection. The
// per-namespace scale envelope now bounds the largest TENANT, not the
// collection, which is what makes a billion-vector multi-tenant collection
// servable on bounded RAM.
//
// Scope fences (MVP, enforced with clear errors): relations, facets, TAMS
// temporal lookup, and vector-space CRUD are not yet routed for partitioned
// collections; partition keys must be kebab-case strings; the partition
// field is immutable after create.

use crate::models::{FilterValue, IngestChunk, MetadataValue};
use std::collections::HashMap;

/// Separator between parent collection name and partition value in the
/// internal namespace. User-facing collection names must not contain it
/// (enforced at create); it is otherwise valid kebab-case, so every existing
/// storage/path rule accepts partition namespaces unchanged.
pub const PART_SEP: &str = "--part--";

/// Max partitions a single set-membership search may fan out to.
pub const MAX_SEARCH_FANOUT: usize = 16;

/// Internal namespace for one partition of a parent collection.
pub fn partition_ns(parent: &str, pval: &str) -> String {
    format!("{parent}{PART_SEP}{pval}")
}

/// Is this namespace a partition (vs a user-facing collection)?
pub fn is_partition_ns(ns: &str) -> bool {
    ns.contains(PART_SEP)
}

/// The parent collection of a partition namespace, or None for a normal one.
pub fn parent_of(ns: &str) -> Option<&str> {
    ns.split_once(PART_SEP).map(|(parent, _)| parent)
}

/// The namespace whose id allocator a collection mints from: partitions share
/// the PARENT's allocator so chunk ids are unique across the whole collection
/// (delete-by-id and search results would otherwise be ambiguous).
pub fn alloc_ns(ns: &str) -> &str {
    parent_of(ns).unwrap_or(ns)
}

/// Partition values become path/namespace segments — hold them to the same
/// kebab-case rule as collection names, and bound the length so a hostile
/// value can't manufacture absurd object keys.
pub fn validate_partition_value(v: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if v.is_empty() || v.len() > 64 || !v.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err(format!(
            "partition value '{v}' is invalid: use letters, digits, and hyphens (max 64 chars)"
        )
        .into());
    }
    if v.contains(PART_SEP) {
        return Err(format!("partition value '{v}' must not contain '{PART_SEP}'").into());
    }
    Ok(())
}

/// Group an ingest batch by its partition value, validating that every chunk
/// carries a usable `metadata[field]` string.
pub fn group_by_partition(
    field: &str,
    chunks: Vec<IngestChunk>,
) -> Result<HashMap<String, Vec<IngestChunk>>, Box<dyn std::error::Error + Send + Sync>> {
    let mut groups: HashMap<String, Vec<IngestChunk>> = HashMap::new();
    for chunk in chunks {
        let pval = match chunk.metadata.get(field) {
            Some(MetadataValue::String(s)) => s.clone(),
            Some(_) => {
                return Err(format!(
                    "chunk '{}': partition field '{field}' must be a string",
                    chunk.file_id
                )
                .into());
            }
            None => {
                return Err(format!(
                    "chunk '{}' is missing partition field '{field}' \
                     (this collection is partitioned by it)",
                    chunk.file_id
                )
                .into());
            }
        };
        validate_partition_value(&pval)?;
        groups.entry(pval).or_default().push(chunk);
    }
    Ok(groups)
}

/// Resolve which partition values a filtered request targets. Exact match
/// routes to one partition; set membership (`in`) fans out (capped). Anything
/// else is an error — a partitioned collection cannot be scanned blind.
pub fn partition_values_from_filters(
    field: &str,
    filters: &HashMap<String, FilterValue>,
) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
    let missing = || -> Box<dyn std::error::Error + Send + Sync> {
        format!(
            "this collection is partitioned by '{field}': include filters.{field} \
             (exact value, or {{\"in\": [...]}} for up to {MAX_SEARCH_FANOUT} partitions)"
        )
        .into()
    };
    let values = match filters.get(field) {
        Some(FilterValue::Exact(MetadataValue::String(s))) => vec![s.clone()],
        Some(FilterValue::Condition(cond)) => match &cond.in_values {
            Some(vs) if !vs.is_empty() => vs.clone(),
            _ => return Err(missing()),
        },
        Some(_) => return Err(missing()),
        None => return Err(missing()),
    };
    if values.len() > MAX_SEARCH_FANOUT {
        return Err(format!(
            "filters.{field} names {} partitions; max fan-out is {MAX_SEARCH_FANOUT}",
            values.len()
        )
        .into());
    }
    for v in &values {
        validate_partition_value(v)?;
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::FilterCondition;

    #[test]
    fn namespace_roundtrip() {
        let ns = partition_ns("videos", "acme");
        assert_eq!(ns, "videos--part--acme");
        assert!(is_partition_ns(&ns));
        assert_eq!(parent_of(&ns), Some("videos"));
        assert_eq!(alloc_ns(&ns), "videos");
        assert!(!is_partition_ns("videos"));
        assert_eq!(parent_of("videos"), None);
        assert_eq!(alloc_ns("videos"), "videos");
    }

    #[test]
    fn partition_value_rules() {
        assert!(validate_partition_value("acme-01").is_ok());
        assert!(validate_partition_value("").is_err());
        assert!(validate_partition_value("has space").is_err());
        assert!(validate_partition_value("a--part--b").is_err());
        assert!(validate_partition_value(&"x".repeat(65)).is_err());
    }

    #[test]
    fn filter_routing() {
        let field = "tenant";
        let mut f = HashMap::new();
        assert!(partition_values_from_filters(field, &f).is_err());

        f.insert(
            field.to_string(),
            FilterValue::Exact(MetadataValue::String("acme".into())),
        );
        assert_eq!(
            partition_values_from_filters(field, &f).unwrap(),
            vec!["acme".to_string()]
        );

        f.insert(
            field.to_string(),
            FilterValue::Condition(FilterCondition {
                gte: None,
                lte: None,
                contains: None,
                in_values: Some(vec!["a".into(), "b".into()]),
            }),
        );
        assert_eq!(partition_values_from_filters(field, &f).unwrap().len(), 2);

        let too_many: Vec<String> = (0..MAX_SEARCH_FANOUT + 1)
            .map(|i| format!("t{i}"))
            .collect();
        f.insert(
            field.to_string(),
            FilterValue::Condition(FilterCondition {
                gte: None,
                lte: None,
                contains: None,
                in_values: Some(too_many),
            }),
        );
        assert!(partition_values_from_filters(field, &f).is_err());
    }
}
