// search/filter_pushdown.rs — Internal filter expression + compiler.
//
// v0 scope: conjunction (AND) of equality and range predicates.
// OR / NOT / nested boolean are deferred to v0.1.
//
// The public API surface lives in `models::FilterValue` (a per-field map). This
// module compiles that map into an internal `FilterExpr` that is cheap to
// evaluate against a chunk's metadata and to plan against a `FilterIndex`.

use std::collections::HashMap;

use crate::models::{FilterCondition, FilterValue, MetadataValue};

/// Internal filter expression. v0 is a flat AND of predicates.
#[derive(Debug, Clone, Default)]
pub struct FilterExpr {
    pub predicates: Vec<Predicate>,
}

#[derive(Debug, Clone)]
pub enum Predicate {
    /// `field == value` using typed metadata equality.
    Eq { field: String, value: MetadataValue },
    /// `gte <= field <= lte`. Either bound may be absent.
    Range {
        field: String,
        gte: Option<f64>,
        lte: Option<f64>,
    },
    /// `value IN field` for StringList metadata.
    Contains { field: String, value: String },
    /// `field IN {values...}` (set membership on the string form).
    In { field: String, values: Vec<String> },
}

impl FilterExpr {
    pub fn is_empty(&self) -> bool {
        self.predicates.is_empty()
    }

    /// Compile a `filters` map (the public API shape) into the internal expression.
    /// Plain values become equality predicates; conditions become range / contains / in.
    pub fn compile(filters: &HashMap<String, FilterValue>) -> Self {
        let mut predicates = Vec::with_capacity(filters.len());
        for (field, value) in filters {
            match value {
                FilterValue::Exact(mv) => {
                    predicates.push(Predicate::Eq {
                        field: field.clone(),
                        value: mv.clone(),
                    });
                }
                FilterValue::Condition(cond) => push_condition(&mut predicates, field, cond),
            }
        }
        FilterExpr { predicates }
    }
}

fn push_condition(out: &mut Vec<Predicate>, field: &str, cond: &FilterCondition) {
    if cond.gte.is_some() || cond.lte.is_some() {
        out.push(Predicate::Range {
            field: field.to_string(),
            gte: cond.gte,
            lte: cond.lte,
        });
    }
    if let Some(v) = &cond.contains {
        out.push(Predicate::Contains {
            field: field.to_string(),
            value: v.clone(),
        });
    }
    if let Some(values) = &cond.in_values {
        out.push(Predicate::In {
            field: field.to_string(),
            values: values.clone(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Semantics (eq / range / contains / in, AND across fields) are covered
    // end-to-end in filter_index.rs tests via FilterIndex::eligible — the one
    // live evaluator. These only pin the compile() shape.
    #[test]
    fn compile_shapes() {
        let mut f = HashMap::new();
        f.insert(
            "org_id".into(),
            FilterValue::Exact(MetadataValue::String("acme".into())),
        );
        f.insert(
            "created_at".into(),
            FilterValue::Condition(FilterCondition {
                gte: Some(100.0),
                lte: Some(200.0),
                contains: None,
                in_values: None,
            }),
        );
        let expr = FilterExpr::compile(&f);
        assert_eq!(expr.predicates.len(), 2);
        assert!(!expr.is_empty());
        assert!(FilterExpr::compile(&HashMap::new()).is_empty());
    }
}
