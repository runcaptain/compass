// search/ivf.rs — IVF (inverted-file) clustering for serve-from-storage.
//
// Compaction k-means-clusters each vector space and writes two sections into
// the segment: `cent:<space>` (tiny: centroids + a cluster directory) and
// `clu:<space>` (the vectors, grouped by cluster). A cold query then needs
// only: the centroids (cached, a few hundred KB), and range-GETs of the
// `nprobe` nearest clusters — instead of materializing the whole segment.
// That is what turns "attach = rebuild everything" into "query = a handful
// of small reads": the difference between warm and true serverless.
//
// Section formats (all little-endian):
//   cent:<space> = [u32 k][u32 dims]
//                  [k × dims × f32 centroids]
//                  [k × (u64 offset, u64 len, u32 count)]   cluster directory,
//                  offsets relative to the START of clu:<space>'s body
//   clu:<space>  = concatenation of clusters, each [count × (u64 id, dims×f32)]
//
// Vectors are stored L2-NORMALIZED in `clu` so scoring is a plain dot
// product (cosine == dot on unit vectors); centroids are means of normalized
// vectors, re-normalized.

/// Below this many rows a space is stored as the flat `emb:` section and cold
/// queries brute-force it — clustering tiny sets costs more than it saves.
pub const CLUSTER_MIN_ROWS: usize = 5_000;

/// Cap on k-means training sample: training cost is O(sample × k × dims);
/// assignment of ALL rows is a single pass afterwards.
const TRAIN_SAMPLE_MAX: usize = 20_000;
const KMEANS_ITERS: usize = 8;

/// Number of clusters for n rows: sqrt(n), clamped. At 5M rows and 384 dims
/// this keeps a cluster ~3.5MB — a few parallel range-GETs per query.
pub fn cluster_count(n: usize) -> usize {
    ((n as f64).sqrt() as usize).clamp(16, 4096)
}

#[inline]
pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// K-means (Lloyd) over normalized vectors, trained on a deterministic
/// sample. Returns (centroids, assignment of EVERY input row).
/// Deterministic: seeded by row order, no RNG.
pub fn kmeans(rows: &[(u64, Vec<f32>)], dims: usize, k: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
    let n = rows.len();
    let k = k.min(n).max(1);

    // Deterministic training sample: evenly-strided rows.
    let stride = (n / TRAIN_SAMPLE_MAX).max(1);
    let sample: Vec<&[f32]> = rows
        .iter()
        .step_by(stride)
        .map(|(_, v)| v.as_slice())
        .collect();

    // Init: evenly-strided sample points as seeds.
    let seed_stride = (sample.len() / k).max(1);
    let mut centroids: Vec<Vec<f32>> = sample
        .iter()
        .step_by(seed_stride)
        .take(k)
        .map(|v| v.to_vec())
        .collect();
    while centroids.len() < k {
        centroids.push(centroids[centroids.len() % sample.len().max(1)].clone());
    }

    let nearest = |cents: &[Vec<f32>], v: &[f32]| -> usize {
        let mut best = 0usize;
        let mut best_d = f32::MIN;
        for (i, c) in cents.iter().enumerate() {
            let d = dot(c, v); // unit vectors: max dot == min angle
            if d > best_d {
                best_d = d;
                best = i;
            }
        }
        best
    };

    for _ in 0..KMEANS_ITERS {
        let mut sums = vec![vec![0f32; dims]; k];
        let mut counts = vec![0usize; k];
        for v in &sample {
            let c = nearest(&centroids, v);
            for (s, x) in sums[c].iter_mut().zip(v.iter()) {
                *s += x;
            }
            counts[c] += 1;
        }
        for (i, (sum, cnt)) in sums.iter_mut().zip(counts.iter()).enumerate() {
            if *cnt > 0 {
                for x in sum.iter_mut() {
                    *x /= *cnt as f32;
                }
                normalize(sum);
                centroids[i] = std::mem::take(sum);
            }
            // Empty cluster: keep the old centroid (harmless; directory entry
            // just ends up with count 0).
        }
    }

    let assignment: Vec<usize> = rows.iter().map(|(_, v)| nearest(&centroids, v)).collect();
    (centroids, assignment)
}

/// Directory entry for one cluster inside `clu:<space>`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterRef {
    pub offset: u64,
    pub len: u64,
    pub count: u32,
}

/// Build the `cent:<space>` and `clu:<space>` section bodies for one space.
/// Input rows may be un-normalized; they are normalized in place here.
pub fn build_sections(mut rows: Vec<(u64, Vec<f32>)>, dims: usize) -> (Vec<u8>, Vec<u8>) {
    for (_, v) in rows.iter_mut() {
        normalize(v);
    }
    let k = cluster_count(rows.len());
    let (centroids, assignment) = kmeans(&rows, dims, k);

    // Group row indexes by cluster, then lay clusters out contiguously.
    let mut by_cluster: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (row_idx, &c) in assignment.iter().enumerate() {
        by_cluster[c].push(row_idx);
    }

    let row_size = 8 + dims * 4;
    let mut clu = Vec::with_capacity(rows.len() * row_size);
    let mut dir: Vec<ClusterRef> = Vec::with_capacity(k);
    for members in &by_cluster {
        let offset = clu.len() as u64;
        for &ri in members {
            let (id, v) = &rows[ri];
            clu.extend_from_slice(&id.to_le_bytes());
            for x in v {
                clu.extend_from_slice(&x.to_le_bytes());
            }
        }
        dir.push(ClusterRef {
            offset,
            len: (members.len() * row_size) as u64,
            count: members.len() as u32,
        });
    }

    let mut cent = Vec::with_capacity(8 + k * dims * 4 + k * 20);
    cent.extend_from_slice(&(k as u32).to_le_bytes());
    cent.extend_from_slice(&(dims as u32).to_le_bytes());
    for c in &centroids {
        for x in c {
            cent.extend_from_slice(&x.to_le_bytes());
        }
    }
    for d in &dir {
        cent.extend_from_slice(&d.offset.to_le_bytes());
        cent.extend_from_slice(&d.len.to_le_bytes());
        cent.extend_from_slice(&d.count.to_le_bytes());
    }
    (cent, clu)
}

/// Parsed `cent:<space>` section.
pub struct Centroids {
    pub dims: usize,
    pub centroids: Vec<Vec<f32>>,
    pub dir: Vec<ClusterRef>,
}

pub fn parse_cent(body: &[u8]) -> Option<Centroids> {
    if body.len() < 8 {
        return None;
    }
    let k = u32::from_le_bytes(body[0..4].try_into().ok()?) as usize;
    let dims = u32::from_le_bytes(body[4..8].try_into().ok()?) as usize;
    let cent_bytes = k.checked_mul(dims)?.checked_mul(4)?;
    let dir_bytes = k.checked_mul(20)?;
    if body.len() < 8 + cent_bytes + dir_bytes {
        return None;
    }
    let mut centroids = Vec::with_capacity(k);
    let mut pos = 8;
    for _ in 0..k {
        let mut v = Vec::with_capacity(dims);
        for _ in 0..dims {
            v.push(f32::from_le_bytes(body[pos..pos + 4].try_into().ok()?));
            pos += 4;
        }
        centroids.push(v);
    }
    let mut dir = Vec::with_capacity(k);
    for _ in 0..k {
        let offset = u64::from_le_bytes(body[pos..pos + 8].try_into().ok()?);
        let len = u64::from_le_bytes(body[pos + 8..pos + 16].try_into().ok()?);
        let count = u32::from_le_bytes(body[pos + 16..pos + 20].try_into().ok()?);
        pos += 20;
        dir.push(ClusterRef { offset, len, count });
    }
    Some(Centroids {
        dims,
        centroids,
        dir,
    })
}

/// Iterate `(id, vector)` rows out of a cluster blob.
pub fn parse_cluster_rows(body: &[u8], dims: usize) -> impl Iterator<Item = (u64, Vec<f32>)> + '_ {
    let row = 8 + dims * 4;
    body.chunks_exact(row).map(move |r| {
        let id = u64::from_le_bytes(r[0..8].try_into().unwrap());
        let mut v = Vec::with_capacity(dims);
        for d in 0..dims {
            let o = 8 + d * 4;
            v.push(f32::from_le_bytes(r[o..o + 4].try_into().unwrap()));
        }
        (id, v)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(n: usize, dims: usize) -> Vec<(u64, Vec<f32>)> {
        // Deterministic, mildly clustered data: 8 anchor directions + noise.
        (0..n)
            .map(|i| {
                let anchor = i % 8;
                let v: Vec<f32> = (0..dims)
                    .map(|d| {
                        let base = if d % 8 == anchor { 1.0 } else { 0.1 };
                        base + ((i * 31 + d * 17) % 97) as f32 / 970.0
                    })
                    .collect();
                (i as u64, v)
            })
            .collect()
    }

    #[test]
    fn sections_roundtrip_and_cover_all_rows() {
        let dims = 16;
        let rows = synthetic(1000, dims);
        let (cent, clu) = build_sections(rows.clone(), dims);
        let parsed = parse_cent(&cent).expect("cent parses");
        assert_eq!(parsed.dims, dims);
        let total: u32 = parsed.dir.iter().map(|d| d.count).sum();
        assert_eq!(total as usize, rows.len(), "every row lands in a cluster");
        // Every directory range decodes to exactly `count` rows and all ids
        // survive.
        let mut seen = std::collections::HashSet::new();
        for d in &parsed.dir {
            let body = &clu[d.offset as usize..(d.offset + d.len) as usize];
            let rows: Vec<_> = parse_cluster_rows(body, dims).collect();
            assert_eq!(rows.len(), d.count as usize);
            for (id, v) in rows {
                assert_eq!(v.len(), dims);
                assert!(seen.insert(id), "id {id} duplicated across clusters");
            }
        }
        assert_eq!(seen.len(), 1000);
    }

    #[test]
    fn nearest_cluster_probe_finds_exact_vector() {
        // Self-recall: probing the nearest clusters for a vector that IS in
        // the index must find it with a modest nprobe.
        let dims = 16;
        let rows = synthetic(2000, dims);
        let (cent, clu) = build_sections(rows.clone(), dims);
        let parsed = parse_cent(&cent).unwrap();

        let mut hits = 0;
        let probes = 4;
        for probe_i in (0..2000).step_by(97) {
            let mut q = rows[probe_i].1.clone();
            normalize(&mut q);
            let mut ranked: Vec<(usize, f32)> = parsed
                .centroids
                .iter()
                .enumerate()
                .map(|(i, c)| (i, dot(c, &q)))
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let found = ranked.iter().take(probes).any(|(ci, _)| {
                let d = parsed.dir[*ci];
                let body = &clu[d.offset as usize..(d.offset + d.len) as usize];
                parse_cluster_rows(body, dims).any(|(id, _)| id == rows[probe_i].0)
            });
            if found {
                hits += 1;
            }
        }
        let total = (0..2000).step_by(97).count();
        assert!(
            hits * 10 >= total * 9,
            "self-recall with nprobe={probes}: {hits}/{total}"
        );
    }
}
