#!/usr/bin/env bash
# Full-surface E2E against a LIVE stack: every endpoint, every filter operator,
# every role behavior. Usage: FULL=host:port WRITER=host:port ./scripts/e2e.sh
set -u
FULL=${FULL:-localhost:4001}
WRITER=${WRITER:-localhost:4009}
COLD=${COLD:-}   # optional: a COMPASS_COLD_SERVE node against the same bucket
pass=0; fail=0
ok(){ echo "  ✅ $1"; pass=$((pass+1)); }
bad(){ echo "  ❌ $1 ($2)"; fail=$((fail+1)); }
jqn(){ python3 -c "import sys,json;d=json.load(sys.stdin);print($1)" 2>/dev/null; }
post(){ curl -s -X POST "$1" -H 'content-type: application/json' -d "$2"; }

echo "── health + metrics ──"
[ "$(curl -s $FULL/health | jqn "d['status']")" = "ok" ] && ok health || bad health x
curl -s $FULL/metrics | grep -q compass_search_requests_total && ok metrics || bad metrics x

echo "── collections CRUD ──"
post $FULL/collections '{"name":"e2e","embedding_dims":4}' >/dev/null
[ "$(curl -s $FULL/collections/e2e | jqn "d['name']")" = "e2e" ] && ok "create+get" || bad create x
curl -s $FULL/collections | grep -q '"e2e"' && ok list || bad list x
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST $FULL/collections -H 'content-type: application/json' -d '{"name":"e2e"}')
[ "$code" -ge 400 ] && ok "duplicate create rejected" || bad dup "$code"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST $FULL/collections -H 'content-type: application/json' -d '{"name":"Bad Name!"}')
[ "$code" -ge 400 ] && ok "invalid name rejected" || bad name "$code"

echo "── ingest: hierarchy, client refs, metadata types ──"
r=$(post $FULL/collections/e2e/ingest '{"chunks":[
 {"client_id":"src1","file_id":"v1","chunk_index":0,"doc_type":"source","text":"Premier League match Arsenal Chelsea","metadata":{"kind":"video","priority":5,"active":true,"tags":["sports","football"],"created_at":"2026-07-01T00:00:00Z"},"embeddings":{"default":[0.9,0.1,0.1,0.1]}},
 {"client_id":"seg1","file_id":"s1","chunk_index":0,"doc_type":"segment","parent_ref":"src1","group_id":"src1","text":"goal celebration minute 34","metadata":{"timerange_start_ms":2040000,"timerange_end_ms":2055000,"priority":9},"embeddings":{"default":[0.1,0.9,0.1,0.1]}},
 {"client_id":"seg2","file_id":"s2","chunk_index":0,"doc_type":"segment","parent_ref":"src1","group_id":"src1","text":"halftime interview coach","metadata":{"timerange_start_ms":2700000,"timerange_end_ms":2760000,"priority":2},"embeddings":{"default":[0.1,0.1,0.9,0.1]}}]}')
n=$(echo "$r" | jqn "d['indexed']"); seq0=$(echo "$r" | jqn "d.get('seq')")
[ "$n" = "3" ] && ok "ingest 3 (hierarchy via parent_ref)" || bad ingest "$n"
[ "$seq0" != "None" ] && ok "ingest returns seq (cloud)" || bad seq x
id_src=$(echo "$r" | jqn "d['id_map']['src1']"); id_seg1=$(echo "$r" | jqn "d['id_map']['seg1']"); id_seg2=$(echo "$r" | jqn "d['id_map']['seg2']")
r=$(post $FULL/collections/e2e/ingest '{"chunks":[{"file_id":"legacy","chunk_index":0,"text":"legacy embedding field","embedding":[0.5,0.5,0.5,0.5]}]}')
[ "$(echo "$r" | jqn "d['indexed']")" = "1" ] && ok "legacy single-embedding field" || bad legacy x
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST $FULL/collections/e2e/ingest -H 'content-type: application/json' -d '{"chunks":[{"file_id":"bad","chunk_index":0,"text":"x","embeddings":{"default":[0.1,0.2]}}]}')
[ "$code" -ge 400 ] && ok "wrong-dims embedding rejected" || bad dims "$code"

echo "── search: modes, filters, scoring, explain ──"
n=$(post $FULL/collections/e2e/search '{"query":"goal celebration","mode":"fts","top_k":5}' | jqn "len(d['results'])")
[ "$n" -ge 1 ] && ok "fts" || bad fts "$n"
n=$(post $FULL/collections/e2e/search '{"query":"","mode":"semantic","query_vector":[0.1,0.9,0.1,0.1],"top_k":1}' | jqn "d['results'][0]['chunk']['file_id']")
[ "$n" = "s1" ] && ok "semantic nearest" || bad semantic "$n"
n=$(post $FULL/collections/e2e/search '{"query":"goal","mode":"hybrid","query_vector":[0.1,0.9,0.1,0.1],"top_k":5,"score_weights":{"rrf_k":60.0,"fts_weight":2.0,"semantic_weight":0.5}}' | jqn "len(d['results'])")
[ "$n" -ge 1 ] && ok "hybrid + score_weights" || bad hybrid "$n"
n=$(post $FULL/collections/e2e/search '{"query":"","mode":"semantic","query_vector":[0.5,0.5,0.5,0.5],"top_k":10,"filters":{"kind":"video"}}' | jqn "len(d['results'])")
[ "$n" = "1" ] && ok "filter: exact string" || bad f-eq "$n"
n=$(post $FULL/collections/e2e/search '{"query":"","mode":"semantic","query_vector":[0.5,0.5,0.5,0.5],"top_k":10,"filters":{"priority":{"gte":3,"lte":10}}}' | jqn "len(d['results'])")
[ "$n" = "2" ] && ok "filter: numeric range" || bad f-range "$n"
n=$(post $FULL/collections/e2e/search '{"query":"","mode":"semantic","query_vector":[0.5,0.5,0.5,0.5],"top_k":10,"filters":{"tags":{"contains":"sports"}}}' | jqn "len(d['results'])")
[ "$n" = "1" ] && ok "filter: array contains" || bad f-contains "$n"
n=$(post $FULL/collections/e2e/search '{"query":"","mode":"semantic","query_vector":[0.5,0.5,0.5,0.5],"top_k":10,"filters":{"doc_type":{"in":["segment"]}}}' | jqn "len(d['results'])")
[ "$n" = "2" ] && ok "filter: set membership (doc_type mirror)" || bad f-in "$n"
n=$(post $FULL/collections/e2e/search '{"query":"","mode":"semantic","query_vector":[0.5,0.5,0.5,0.5],"top_k":10,"filters":{"active":true}}' | jqn "len(d['results'])")
[ "$n" = "1" ] && ok "filter: bool" || bad f-bool "$n"
r=$(post $FULL/collections/e2e/search '{"query":"match","mode":"semantic","query_vector":[0.9,0.1,0.1,0.1],"top_k":5,"filters":{"kind":"video"},"explain":true}')
[ "$(echo "$r" | jqn "d['explain']['filter']['eligible_count']")" = "1" ] && ok "explain plan" || bad explain x
n=$(post $FULL/collections/e2e/search '{"query":"goal","mode":"fts","top_k":5,"recency_preset":"mild","recency_field":"created_at"}' | jqn "len(d['results'])")
[ "$n" -ge 1 ] && ok "recency preset" || bad recency "$n"
n=$(post $FULL/collections/e2e/search '{"query":"interview","mode":"fts","top_k":5,"boosts":[{"field":"priority","gte":3,"weight":2.0}],"relationship_boost":{"parent_weight":0.3,"sibling_weight":0.1}}' | jqn "len(d['results'])")
[ "$n" -ge 1 ] && ok "boosts + relationship_boost" || bad boosts "$n"

echo "── relations ──"
r=$(post $FULL/collections/e2e/relations "{\"relations\":[{\"source_chunk_id\":$id_seg1,\"target_chunk_id\":$id_seg2,\"relation_type\":\"follows\"}]}")
rid=$(echo "$r" | jqn "d['relations'][0]['relation_id']")
[ -n "$rid" ] && ok "create relation" || bad rel x
n=$(curl -s "$FULL/collections/e2e/chunks/$id_seg1/relations?direction=outgoing&types=follows" | jqn "d['total']")
[ "$n" = "1" ] && ok "list relations (direction+type)" || bad rel-list "$n"
st=$(curl -s "$FULL/collections/e2e/chunks/$id_seg1/relations" | jqn "d['relations'][0]['target_status']")
[ "$st" = "found" ] && ok "target_status resolution" || bad status "$st"
n=$(post $FULL/collections/e2e/search '{"query":"goal","mode":"fts","top_k":5,"include_relations":true}' | jqn "d['results'][0].get('relations') is not None")
[ "$n" = "True" ] && ok "include_relations in search" || bad inc-rel "$n"
code=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE $FULL/collections/e2e/relations/$rid)
[ "$code" = "204" ] && ok "delete relation" || bad rel-del "$code"

echo "── facets + TAMS ──"
curl -s "$FULL/collections/e2e/facets" | grep -q "video" && ok facets || bad facets x
n=$(curl -s "$FULL/collections/e2e/segments/at?asset=src1&time_ms=2050000" | jqn "len(d.get('segments',d.get('results',[])))")
[ "$n" -ge 1 ] && ok "TAMS point lookup" || bad tams "$n"

echo "── vector spaces ──"
post $FULL/collections/e2e/vector-spaces '{"name":"wide","dims":8,"model":"test"}' >/dev/null
curl -s $FULL/collections/e2e/vector-spaces | grep -q wide && ok "add+list space" || bad vs x
r=$(post $FULL/collections/e2e/ingest '{"chunks":[{"file_id":"w","chunk_index":0,"text":"wide vec","embeddings":{"default":[0.2,0.2,0.2,0.2],"wide":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]}}]}')
[ "$(echo "$r" | jqn "d['indexed']")" = "1" ] && ok "multi-space ingest" || bad ms x
n=$(post $FULL/collections/e2e/search '{"query":"","mode":"semantic","vector_space":"wide","query_vector":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],"top_k":1}' | jqn "len(d['results'])")
[ "$n" = "1" ] && ok "search named space" || bad ms-search "$n"
code=$(curl -s -o /dev/null -w '%{http_code}' -X PUT $FULL/collections/e2e/default-vector-space -H 'content-type: application/json' -d '{"name":"wide"}')
[ "$code" -lt 400 ] && ok "switch default space" || bad def "$code"
curl -s -X PUT $FULL/collections/e2e/default-vector-space -H 'content-type: application/json' -d '{"name":"default"}' >/dev/null
code=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE $FULL/collections/e2e/vector-spaces/wide)
[ "$code" -lt 400 ] && ok "delete space" || bad vs-del "$code"

echo "── writer role ──"
r=$(post $WRITER/collections/e2e/ingest '{"chunks":[{"file_id":"wchunk","chunk_index":0,"text":"from the stateless writer","embeddings":{"default":[0.7,0.7,0.1,0.1]}}]}')
wseq=$(echo "$r" | jqn "d['seq']")
[ "$wseq" != "None" ] && ok "writer ingest returns seq" || bad w-ingest x
n=$(post $FULL/collections/e2e/search "{\"query\":\"\",\"mode\":\"semantic\",\"query_vector\":[0.7,0.7,0.1,0.1],\"top_k\":1,\"min_seq\":$wseq}" | jqn "d['results'][0]['chunk']['file_id']")
[ "$n" = "wchunk" ] && ok "min_seq read-your-writes across nodes" || bad ryw "$n"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST $WRITER/collections/e2e/search -H 'content-type: application/json' -d '{"query":"x"}')
[ "$code" -ge 400 ] && ok "writer refuses reads" || bad w-read "$code"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST $WRITER/collections/ghost/delete -H 'content-type: application/json' -d '{"ids":[1]}')
[ "$code" -ge 400 ] && ok "writer refuses phantom namespace" || bad w-ghost "$code"

echo "── deletes + compact ──"
wid=$(post $FULL/collections/e2e/search '{"query":"","mode":"semantic","query_vector":[0.7,0.7,0.1,0.1],"top_k":1}' | jqn "d['results'][0]['chunk']['id']")
r=$(curl -s -X DELETE $FULL/collections/e2e/chunks/$wid)
[ "$(echo "$r" | jqn "d['deleted']")" = "1" ] && ok "delete by id (+seq $(echo "$r" | jqn "d.get('seq')"))" || bad del x
r=$(post $FULL/collections/e2e/delete '{"filters":{"kind":"video"}}')
[ "$(echo "$r" | jqn "d['deleted']")" = "1" ] && ok "delete by filter" || bad del-f x
n=$(post $FULL/collections/e2e/search '{"query":"","mode":"semantic","query_vector":[0.9,0.1,0.1,0.1],"top_k":10}' | jqn "sum(1 for r in d['results'] if r['chunk']['file_id']=='v1')")
[ "$n" = "0" ] && ok "deleted chunk masked" || bad mask "$n"
r=$(post $FULL/collections/e2e/compact '')
[ -n "$(echo "$r" | jqn "d['compacted_records']")" ] && ok compact || bad compact x
n=$(post $FULL/collections/e2e/search '{"query":"goal","mode":"fts","top_k":5}' | jqn "len(d['results'])")
[ "$n" -ge 1 ] && ok "data survives compaction" || bad post-compact "$n"

echo "── tenant partitions ──"
post $FULL/collections '{"name":"mt","embedding_dims":4,"config":{"partition_by":"tenant"}}' >/dev/null
r=$(post $FULL/collections/mt/ingest '{"chunks":[
 {"client_id":"p1","file_id":"p1","chunk_index":0,"doc_type":"chunk","text":"shared secret alpha","metadata":{"tenant":"acme"},"embeddings":{"default":[0.9,0.1,0.1,0.1]}},
 {"client_id":"p2","file_id":"p2","chunk_index":0,"doc_type":"chunk","text":"shared secret beta","metadata":{"tenant":"globex"},"embeddings":{"default":[0.1,0.9,0.1,0.1]}}]}')
[ "$(echo "$r" | jqn "d['indexed']")" = "2" ] && ok "partitioned ingest routes" || bad p-ingest x
n=$(post $FULL/collections/mt/search '{"query":"secret","mode":"fts","top_k":10,"filters":{"tenant":"acme"}}' | jqn "len(d['results'])")
f1=$(post $FULL/collections/mt/search '{"query":"secret","mode":"fts","top_k":10,"filters":{"tenant":"acme"}}' | jqn "d['results'][0]['chunk']['file_id']")
[ "$n" = "1" ] && [ "$f1" = "p1" ] && ok "tenant isolation (acme sees only its hit)" || bad p-iso "$n/$f1"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST $FULL/collections/mt/search -H 'content-type: application/json' -d '{"query":"secret","mode":"fts"}')
[ "$code" -ge 400 ] && ok "unfiltered partitioned search rejected" || bad p-nofilter "$code"
n=$(post $FULL/collections/mt/search '{"query":"secret","mode":"fts","top_k":10,"filters":{"tenant":{"in":["acme","globex"]}}}' | jqn "len(d['results'])")
[ "$n" = "2" ] && ok "set-membership fan-out merges tenants" || bad p-fanout "$n"
r=$(post $WRITER/collections/mt/ingest '{"chunks":[{"client_id":"p3","file_id":"p3","chunk_index":0,"doc_type":"chunk","text":"writer minted tenant","metadata":{"tenant":"initech"},"embeddings":{"default":[0.1,0.1,0.9,0.1]}}]}')
[ "$(echo "$r" | jqn "d['indexed']")" = "1" ] && ok "writer partitioned ingest" || bad p-writer x
sleep 1
n=$(post $FULL/collections/mt/search '{"query":"minted","mode":"fts","top_k":5,"filters":{"tenant":"initech"}}' | jqn "len(d['results'])")
[ "$n" = "1" ] && ok "writer-minted partition attaches on serving node" || bad p-attach "$n"
r=$(post $FULL/collections/mt/delete '{"filters":{"tenant":"acme"}}')
[ "$(echo "$r" | jqn "d['deleted']")" = "1" ] && ok "partition-scoped delete" || bad p-del x
code=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE $FULL/collections/mt)
[ "$code" -lt 400 ] && ok "partitioned collection cascade delete" || bad p-casc "$code"
curl -s $FULL/collections | grep -q "part--" && bad "partitions hidden from listing" leak || ok "partitions hidden from listing"

echo "── collection delete ──"
code=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE $FULL/collections/e2e)
[ "$code" -lt 400 ] && ok "delete collection" || bad coll-del "$code"
code=$(curl -s -o /dev/null -w '%{http_code}' $FULL/collections/e2e)
[ "$code" = "404" ] || [ "$(curl -s $FULL/collections/e2e)" = "null" ] && ok "collection gone" || bad gone "$code"

if [ -n "$COLD" ]; then
echo "── serve-from-storage (cold node) ──"
post $FULL/collections '{"name":"icy","embedding_dims":4}' >/dev/null
post $FULL/collections/icy/ingest '{"chunks":[
 {"file_id":"i1","chunk_index":0,"doc_type":"chunk","text":"glacier core","metadata":{"kind":"ice"},"embeddings":{"default":[0.9,0.1,0.1,0.1]}},
 {"file_id":"i2","chunk_index":0,"doc_type":"chunk","text":"magma core","metadata":{"kind":"fire"},"embeddings":{"default":[0.1,0.9,0.1,0.1]}}]}' >/dev/null
r=$(post $COLD/collections/icy/search '{"query":"","mode":"semantic","top_k":3,"query_vector":[0.9,0.1,0.1,0.1]}')
f1=$(echo "$r" | jqn "d['results'][0]['chunk']['file_id']")
[ "$f1" = "i1" ] && ok "cold node answers without attach" || bad cold-search "$f1"
n=$(post $COLD/collections/icy/search '{"query":"","mode":"semantic","top_k":3,"query_vector":[0.9,0.1,0.1,0.1],"filters":{"kind":"fire"}}' | jqn "d['results'][0]['chunk']['file_id']")
[ "$n" = "i2" ] && ok "cold filters apply" || bad cold-filter "$n"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST $COLD/collections/icy/search -H 'content-type: application/json' -d '{"query":"glacier","mode":"fts"}')
[ "$code" -ge 400 ] && ok "cold FTS rejected with guidance" || bad cold-fts "$code"
wseq2=$(post $WRITER/collections/icy/ingest '{"chunks":[{"file_id":"i3","chunk_index":0,"doc_type":"chunk","text":"fresh tail","metadata":{"kind":"new"},"embeddings":{"default":[0.1,0.1,0.9,0.1]}}]}' | jqn "d['seq']")
f3=$(post $COLD/collections/icy/search '{"query":"","mode":"semantic","top_k":1,"query_vector":[0.1,0.1,0.9,0.1]}' | jqn "d['results'][0]['chunk']['file_id']")
[ "$f3" = "i3" ] && ok "cold read-your-writes (writer tail visible instantly, seq $wseq2)" || bad cold-ryw "$f3"
curl -s $COLD/metrics | grep -q "compass_cold_searches_total [1-9]" && ok "cold metrics counting" || bad cold-metrics x
curl -s -o /dev/null -X DELETE $FULL/collections/icy
fi

echo ""
echo "E2E RESULT: $pass passed, $fail failed"
[ "$fail" = "0" ]
