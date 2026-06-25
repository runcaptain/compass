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

    /// Evaluate the expression against a chunk's metadata. AND across predicates.
    pub fn eval(&self, metadata: &HashMap<String, MetadataValue>) -> bool {
        self.predicates.iter().all(|p| eval_predicate(p, metadata))
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

fn eval_predicate(p: &Predicate, metadata: &HashMap<String, MetadataValue>) -> bool {
    match p {
        Predicate::Eq { field, value } => match metadata.get(field) {
            Some(mv) => mv == value,
            None => false,
        },
        Predicate::Range { field, gte, lte } => {
            match metadata.get(field).and_then(|m| m.as_f64()) {
                Some(n) => {
                    gte.map(|g| n >= g).unwrap_or(true) && lte.map(|l| n <= l).unwrap_or(true)
                }
                None => false,
            }
        }
        Predicate::Contains { field, value } => match metadata.get(field) {
            Some(MetadataValue::StringList(xs)) => xs.iter().any(|x| x == value),
            Some(MetadataValue::String(s)) => s == value,
            _ => false,
        },
        Predicate::In { field, values } => match metadata.get(field) {
            Some(MetadataValue::String(s)) => values.contains(s),
            None => false,
            _ => false,
        },
    }
}

/// Canonical string form for an equality / set-membership key. Booleans and
/// numbers normalize to a stable string so that the filter index can key on
/// `(field, string)` without juggling typed variants.
pub fn stringify_metadata(mv: &MetadataValue) -> String {
    match mv {
        MetadataValue::Bool(b) => b.to_string(),
        MetadataValue::Int(i) => i.to_string(),
        MetadataValue::Float(f) => f.to_string(),
        MetadataValue::String(s) => s.clone(),
        MetadataValue::StringList(xs) => xs.join(","),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: &[(&str, MetadataValue)]) -> HashMap<String, MetadataValue> {
        pairs
            .iter()
            .cloned()
            .map(|(k, v)| (k.to_string(), v))
            .collect()
    }

    #[test]
    fn eq_matches_string() {
        let mut f = HashMap::new();
        f.insert(
            "org_id".into(),
            FilterValue::Exact(MetadataValue::String("acme".into())),
        );
        let expr = FilterExpr::compile(&f);
        assert!(expr.eval(&meta(&[("org_id", MetadataValue::String("acme".into()))])));
        assert!(!expr.eval(&meta(&[(
            "org_id",
            MetadataValue::String("widgets".into())
        )])));
        assert!(!expr.eval(&meta(&[])));
    }

    #[test]
    fn range_inclusive_bounds() {
        let mut f = HashMap::new();
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
        assert!(expr.eval(&meta(&[("created_at", MetadataValue::Int(100))])));
        assert!(expr.eval(&meta(&[("created_at", MetadataValue::Float(150.5))])));
        assert!(expr.eval(&meta(&[("created_at", MetadataValue::Int(200))])));
        assert!(!expr.eval(&meta(&[("created_at", MetadataValue::Int(99))])));
        assert!(!expr.eval(&meta(&[("created_at", MetadataValue::Int(201))])));
    }

    #[test]
    fn and_of_eq_and_range() {
        let mut f = HashMap::new();
        f.insert(
            "org_id".into(),
            FilterValue::Exact(MetadataValue::String("acme".into())),
        );
        f.insert(
            "created_at".into(),
            FilterValue::Condition(FilterCondition {
                gte: Some(100.0),
                lte: None,
                contains: None,
                in_values: None,
            }),
        );
        let expr = FilterExpr::compile(&f);
        let ok = meta(&[
            ("org_id", MetadataValue::String("acme".into())),
            ("created_at", MetadataValue::Int(150)),
        ]);
        let wrong_org = meta(&[
            ("org_id", MetadataValue::String("widgets".into())),
            ("created_at", MetadataValue::Int(150)),
        ]);
        let too_old = meta(&[
            ("org_id", MetadataValue::String("acme".into())),
            ("created_at", MetadataValue::Int(50)),
        ]);
        assert!(expr.eval(&ok));
        assert!(!expr.eval(&wrong_org));
        assert!(!expr.eval(&too_old));
    }

    #[test]
    fn contains_on_string_list() {
        let mut f = HashMap::new();
        f.insert(
            "tags".into(),
            FilterValue::Condition(FilterCondition {
                gte: None,
                lte: None,
                contains: Some("sports".into()),
                in_values: None,
            }),
        );
        let expr = FilterExpr::compile(&f);
        assert!(expr.eval(&meta(&[(
            "tags",
            MetadataValue::StringList(vec!["sports".into(), "goals".into()]),
        )])));
        assert!(!expr.eval(&meta(&[(
            "tags",
            MetadataValue::StringList(vec!["news".into()]),
        )])));
    }
}
