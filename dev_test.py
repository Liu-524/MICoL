import json
import tqdm
adj_dict = {}
rev_dict = {}
children = set()
with open('data/MAG/taxonomy.txt', 'r') as fin:
    for line in fin:
        split = line.strip().split()
        p = int(split[0])
        c = set([int(x) for x in split[1:]])
        adj_dict[p] = c
        for cc in c:
            rev_dict[cc] = rev_dict.get(cc, set()).union({p})
        children = children.union(c)

for x in adj_dict:
    if (x not in children):
        print(x)
        
w2id = {}
id2w = {}
with open('data/MAG/id2label.txt', 'r') as fin:
    for line in fin:
        split = line.strip().split()
        w2id[split[1]] = int(split[0])
        id2w[int(split[0])] = split[1]
y_gt = []
R = 0
P = 0
a = 0
pa = 0
lp = 0



with open('MAG/MAG_candidates.json', 'r') as fin, open('MAG/MAG_candidates_more.json', 'w') as fout:
    for line in tqdm.tqdm(fin):
        j = json.loads(line)
        pred = set(j['predicted_label'])
        prev = len(pred)
        for x in pred:
        #     pred = pred.union(set([str(s) for s in adj_dict.get(int(x), set())] ))
            pred = pred.union(set([str(s) for s in rev_dict.get(int(x), set())] ))   
        if '41008148' in pred:
            pred.remove('41008148')
        
        lp += len(pred) - prev
        gt = set(j['label'])
        isec = len(pred.intersection(gt))
        R += isec / len(gt)
        if (len(pred) > 0):
            P += isec / len(pred)
            pa += 1
        a += 1
        new_j = dict(j)
        new_j['predicted_label'] = [str(x) for x in pred]
        fout.write(json.dumps(new_j) + '\n')
print(P/pa)
print(R/a)
print(lp)

root = 41008148
level_dict = {}
level_dict[root] = 0
q = [root]
while len(q) > 0:
    curr = q.pop(0)
    if curr in adj_dict:
        for child in adj_dict[curr]:
            level_dict[child] = level_dict[curr] + 1
            q.append(child)
  
#labels = ["60048249", "206134035", "2474386", "165297611", "195324797", "39890363", "70777604", "204321447"]
labels = [["166553842", 3.054604560136795], ["19768560", 2.9081293642520905], ["200065993", 2.5654199719429016], ["146499914", 2.443896174430847], ["105716622", 2.3014424443244934], ["29808475", 2.2686751186847687], ["207648694", 2.2343905866146088], ["206134035", 2.060129463672638], ["70777604", 1.969359815120697], ["114408938", 1.8628124594688416], ["156325763", 1.5025754570960999], ["186644900", 1.2305254340171814], ["60048249", 1.1710509061813354], ["39890363", 1.1533639430999756], ["2474386", 1.1489831805229187], ["2776321320", 1.1394786834716797], ["76482347", 1.0964214205741882], ["26320393", 1.088388867676258], ["195324797", 1.0575013160705566], ["124246873", 1.025088220834732], ["165297611", 1.0186533033847809], ["5655090", 0.6400662139058113], ["9628104", 0.6304876878857613], ["192209626", 0.6092884689569473], ["204321447", 0.5298171043395996], ["5147268", 0.5203781593590975], ["28490314", 0.363495409488678], ["199360897", 0.2682240903377533], ["119857082", 0.2166975885629654], ["11413529", 0.20154716074466705], ["23123220", 0.1038007065653801], ["31972630", 0.022555720061063766], ["44154836", 0.011972332373261452], ["77088390", -0.008851636201143265], ["120314980", -0.01642351783812046], ["31258907", -0.07890462875366211], ["38652104", -0.12667347490787506]]
for lbl, acc in labels:
    print(lbl, level_dict[int(lbl)])
    


histo = [0,0,0,0,0,0]

with open('MAG/MAG_candidates.json', 'r') as fin, open('MAG/MAG_candidates_more.json', 'w') as fout:
    for line in tqdm.tqdm(fin):
        j = json.loads(line)
        gt = j['label']
        for x in gt:
            histo[level_dict[int(x)]] += 1
from matplotlib import pyplot as plt
import numpy as np

print(histo)
print(np.array(histo)/sum(histo))
# while(1):
#     x = input(":")
#     print([id2w[w] for w in adj_dict[w2id[x.strip()]]])