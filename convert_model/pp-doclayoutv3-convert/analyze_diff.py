import json

py = json.load(open(r'd:\code\pp-doclayoutv3-convert\results_python.json', 'r', encoding='utf-8'))
java = json.load(open(r'd:\code\pp-doclayoutv3-convert\results_java.json', 'r', encoding='utf-8'))

for name in ['3820.png', '4051.png']:
    print(f'\n{"="*60}')
    print(f'{name}')
    print(f'{"="*60}')
    
    pd = py[name]['detections']
    jd = java[name]['detections']
    
    print(f'\nPython ({len(pd)} dets):')
    for i, d in enumerate(pd):
        print(f'  [{i+1:2d}] {d["label"]:20s} score={d["score"]:.6f}  box={d["box"]}')
    
    print(f'\nJava ({len(jd)} dets):')
    for i, d in enumerate(jd):
        print(f'  [{i+1:2d}] {d["label"]:20s} score={d["score"]:.6f}  box={d["box"]}')
    
    # Find near-threshold detections (score close to 0.5)
    print(f'\n--- Near-threshold analysis (threshold=0.5) ---')
    all_scores_py = [(d['label'], d['score']) for d in pd]
    all_scores_java = [(d['label'], d['score']) for d in jd]
    
    print('Python scores near threshold:')
    for label, score in all_scores_py:
        if 0.45 < score < 0.55:
            print(f'  {label:20s} score={score:.6f}  delta={score-0.5:+.6f}')
    
    print('Java scores near threshold:')
    for label, score in all_scores_java:
        if 0.45 < score < 0.55:
            print(f'  {label:20s} score={score:.6f}  delta={score-0.5:+.6f}')
