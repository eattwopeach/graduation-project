import re
import json
import glob
from copy import deepcopy
pattern = re.compile('[\W_]+')

comas = sorted(glob.glob('test/results/COMA*.json'))
costs = sorted(glob.glob('test/results/COST*.json'))
for file in comas + costs:
    cor = inv = tot = 0
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            tot += 1
            line = json.loads(line)
            res = [pattern.sub('', i) for i in pattern.sub(' ', line['response']).split('Answer:')[-1].split()]
            for i in res:
                if i in ['A', 'B', 'C', 'D']:
                    break
            if i == ['A', 'B', 'C', 'D'][line['gold']]:
                cor += 1
            elif i not in ['A', 'B', 'C', 'D']:
                if line['choices'][line['gold']].lower() in line['response'].lower():
                    tmp = deepcopy(line['choices'])
                    tmp.pop(line['gold'])
                    flag = True
                    for j in tmp:
                        if j.lower() in line['response'].lower():
                            flag = False
                    if flag:
                        cor += 1
            if 'A' not in res and 'B' not in res and 'C' not in res and 'D' not in res:
                cnt = 0
                for i in line['choices']:
                    if i.lower() in line['response'].lower():
                        cnt += 1
                if cnt != 1:
                    inv += 1
    if tot > 0:
        print(file, round(cor / tot * 100, 2), inv / tot)

csjs = sorted(glob.glob('test/results/CSJ*.json'))
for file in csjs:
    cor = inv = tot = 0
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            tot += 1
            line = json.loads(line)
            right = ['YES', 'Yes', 'Correct', 'Acceptable']
            wrong = ['NO', 'No', 'Incorrect', 'Unacceptable']
            pre = None
            for w in wrong:
                if w in line['response']:
                    pre = 0
            if pre is None:
                for r in right:
                    if r in line['response']:
                        pre = 1
            if pre == int(line['gold']):
                cor += 1
            if pre is None:
                inv += 1
    if tot > 0:
        print(file, round(cor / tot * 100, 2), inv / tot)

post_comas = sorted(glob.glob('test/results/post_COMA*.json'))
post_costs = sorted(glob.glob('test/results/post_COST*.json'))
for file in post_comas + post_costs:
    cor = inv = tot = 0
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            tot += 1
            line = json.loads(line)
            res = [pattern.sub('', i) for i in pattern.sub(' ', line['response']).split('Answer:')[-1].split()]
            for i in res:
                if i in ['A', 'B', 'C', 'D']:
                    break
            if i == ['A', 'B', 'C', 'D'][line['gold']]:
                cor += 1
            elif i not in ['A', 'B', 'C', 'D']:
                if line['choices'][line['gold']].lower() in line['response'].lower():
                    tmp = deepcopy(line['choices'])
                    tmp.pop(line['gold'])
                    flag = True
                    for j in tmp:
                        if j.lower() in line['response'].lower():
                            flag = False
                    if flag:
                        cor += 1
            if 'A' not in res and 'B' not in res and 'C' not in res and 'D' not in res:
                cnt = 0
                for i in line['choices']:
                    if i.lower() in line['response'].lower():
                        cnt += 1
                if cnt != 1:
                    inv += 1
    if tot > 0:
        print(file, round(cor / tot * 100, 2), inv / tot)

post_csjs = sorted(glob.glob('test/results/post_CSJ*.json'))
for file in post_csjs:
    cor = inv = tot = 0
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            tot += 1
            line = json.loads(line)
            right = ['YES', 'Yes', 'Correct', 'Acceptable']
            wrong = ['NO', 'No', 'Incorrect', 'Unacceptable']
            pre = None
            for w in wrong:
                if w in line['response']:
                    pre = 0
            if pre is None:
                for r in right:
                    if r in line['response']:
                        pre = 1
            if pre == int(line['gold']):
                cor += 1
            if pre is None:
                inv += 1
    if tot > 0:
        print(file, round(cor / tot * 100, 2), inv / tot)
