// filter.rs — Post-retrieval metadata filtering with operator support.
//
// Operators:
//   exact match  — "department": "Legal"
//   range        — "timerange_start": {"gte": 2040.0}
//   contains     — "tags": {"contains": "sports"}
//   set member   — "doc_type": {"in": ["segment", "flow"]}
//
// "doc_type" is special-cased: it reads from chunk.doc_type (struct field)
// instead of chunk.metadata, so existing data on disk works without migration.

use crate::models::{DocumentChunk, FilterCondition, FilterValue, MetadataValue};
use std::collections::HashMap;

pub fn matches_filters(chunk: &DocumentChunk, filters: &HashMap<String, FilterValue>) -> bool {
    filters.iter().all(|(key, filter_val)| {
        let meta_val = if key == "doc_type" {
            Some(MetadataValue::String(chunk.doc_type.clone()))
        } else {
            chunk.metadata.get(key).cloned()
        };

        match filter_val {
            FilterValue::Exact(expected) => meta_val.as_ref().map_or(false, |v| v == expected),
            FilterValue::Condition(cond) => eval_condition(meta_val.as_ref(), cond),
        }
    })
}

fn eval_condition(val: Option<&MetadataValue>, cond: &FilterCondition) -> bool {
    if cond.gte.is_some() || cond.lte.is_some() {
        match val.and_then(|v| v.as_f64()) {
            None => return false,
            Some(n) => {
                if let Some(g) = cond.gte {
                    if n < g {
                        return false;
                    }
                }
                if let Some(l) = cond.lte {
                    if n > l {
                        return false;
                    }
                }
            }
        }
    }

    if let Some(ref target) = cond.contains {
        match val {
            Some(MetadataValue::StringList(list)) => {
                if !list.iter().any(|s| s == target) {
                    return false;
                }
            }
            Some(MetadataValue::String(s)) => {
                if s != target {
                    return false;
                }
            }
            _ => return false,
        }
    }

    if let Some(ref allowed) = cond.in_values {
        match val {
            Some(MetadataValue::String(s)) => {
                if !allowed.contains(s) {
                    return false;
                }
            }
            _ => return false,
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::DocumentChunk;
    use std::collections::HashMap;

    fn make_chunk(doc_type: &str, metadata: HashMap<String, MetadataValue>) -> DocumentChunk {
        DocumentChunk {
            id: 1,
            collection: "test".to_string(),
            file_id: "f1".to_string(),
            chunk_index: 0,
            page: None,
            text: "test".to_string(),
            metadata,
            doc_type: doc_type.to_string(),
            parent_id: None,
            group_id: None,
            embeddings: HashMap::new(),
            embedding: None,
        }
    }

    #[test]
    fn exact_match_backward_compat() {
        let mut meta = HashMap::new();
        meta.insert(
            "department".to_string(),
            MetadataValue::String("Legal".to_string()),
        );
        let chunk = make_chunk("chunk", meta);

        let mut filters = HashMap::new();
        filters.insert(
            "department".to_string(),
            FilterValue::Exact(MetadataValue::String("Legal".to_string())),
        );
        assert!(matches_filters(&chunk, &filters));

        filters.insert(
            "department".to_string(),
            FilterValue::Exact(MetadataValue::String("HR".to_string())),
        );
        assert!(!matches_filters(&chunk, &filters));
    }

    #[test]
    fn numeric_range_gte() {
        let mut meta = HashMap::new();
        meta.insert("timerange_start".to_string(), MetadataValue::Float(2040.0));
        let chunk = make_chunk("segment", meta);

        let mut filters = HashMap::new();
        filters.insert(
            "timerange_start".to_string(),
            FilterValue::Condition(FilterCondition {
                gte: Some(2000.0),
                lte: None,
                contains: None,
                in_values: None,
            }),
        );
        assert!(matches_filters(&chunk, &filters));

        filters.insert(
            "timerange_start".to_string(),
            FilterValue::Condition(FilterCondition {
                gte: Some(2100.0),
                lte: None,
                contains: None,
                in_values: None,
            }),
        );
        assert!(!matches_filters(&chunk, &filters));
    }

    #[test]
    fn numeric_range_combined() {
        let mut meta = HashMap::new();
        meta.insert("priority".to_string(), MetadataValue::Int(5));
        let chunk = make_chunk("chunk", meta);

        let mut filters = HashMap::new();
        filters.insert(
            "priority".to_string(),
            FilterValue::Condition(FilterCondition {
                gte: Some(3.0),
                lte: Some(10.0),
                contains: None,
                in_values: None,
            }),
        );
        assert!(matches_filters(&chunk, &filters));

        filters.insert(
            "priority".to_string(),
            FilterValue::Condition(FilterCondition {
                gte: Some(6.0),
                lte: Some(10.0),
                contains: None,
                in_values: None,
            }),
        );
        assert!(!matches_filters(&chunk, &filters));
    }

    #[test]
    fn contains_string_list() {
        let mut meta = HashMap::new();
        meta.insert(
            "tags".to_string(),
            MetadataValue::StringList(vec!["sports".to_string(), "goals".to_string()]),
        );
        let chunk = make_chunk("chunk", meta);

        let mut filters = HashMap::new();
        filters.insert(
            "tags".to_string(),
            FilterValue::Condition(FilterCondition {
                gte: None,
                lte: None,
                contains: Some("sports".to_string()),
                in_values: None,
            }),
        );
        assert!(matches_filters(&chunk, &filters));

        filters.insert(
            "tags".to_string(),
            FilterValue::Condition(FilterCondition {
                gte: None,
                lte: None,
                contains: Some("music".to_string()),
                in_values: None,
            }),
        );
        assert!(!matches_filters(&chunk, &filters));
    }

    #[test]
    fn in_set_membership() {
        let chunk = make_chunk("segment", HashMap::new());

        let mut filters = HashMap::new();
        filters.insert(
            "doc_type".to_string(),
            FilterValue::Condition(FilterCondition {
                gte: None,
                lte: None,
                contains: None,
                in_values: Some(vec!["segment".to_string(), "flow".to_string()]),
            }),
        );
        assert!(matches_filters(&chunk, &filters));

        filters.insert(
            "doc_type".to_string(),
            FilterValue::Condition(FilterCondition {
                gte: None,
                lte: None,
                contains: None,
                in_values: Some(vec!["source".to_string()]),
            }),
        );
        assert!(!matches_filters(&chunk, &filters));
    }

    #[test]
    fn doc_type_special_case() {
        let chunk = make_chunk("segment", HashMap::new());

        let mut filters = HashMap::new();
        filters.insert(
            "doc_type".to_string(),
            FilterValue::Exact(MetadataValue::String("segment".to_string())),
        );
        assert!(matches_filters(&chunk, &filters));
    }

    #[test]
    fn missing_field_returns_false() {
        let chunk = make_chunk("chunk", HashMap::new());

        let mut filters = HashMap::new();
        filters.insert(
            "nonexistent".to_string(),
            FilterValue::Exact(MetadataValue::String("value".to_string())),
        );
        assert!(!matches_filters(&chunk, &filters));
    }
}
