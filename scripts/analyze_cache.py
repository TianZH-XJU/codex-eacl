import json, sys
p=sys.argv[1]
rows=[json.loads(x) for x in open(p,encoding='utf-8')]
for r in sorted(rows,key=lambda x:-x['latency_ms'])[:10]:
    print(r['index'], r['strategy'], r['latency_ms'], r['correct'], r['text'][:50])
print('count', len(rows))