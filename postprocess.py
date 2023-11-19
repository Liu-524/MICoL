import json
import argparse
import os
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--architecture', required=True, type=str)
args = parser.parse_args()

pred = []
with open(os.path.join(args.output_dir, f'prediction_{args.architecture}.txt')) as fin:
	for line in fin:
		data = float(line.strip())
		pred.append(data)

adj_dict = {}
rev_dict = {}
children = set()
with open(f'data/{args.dataset}/taxonomy.txt', 'r') as fin:
    for line in fin:
        split = line.strip().split()
        p = int(split[0])
        c = set([int(x) for x in split[1:]])
        adj_dict[p] = c
        for cc in c:
            rev_dict[cc] = rev_dict.get(cc, set()).union({p})
        children = children.union(c)
    


i = 0
with open(f'{args.dataset}/{args.dataset}_candidates.json') as fin, \
    open(os.path.join(args.output_dir, f'prediction_{args.architecture}.json'), 'w') as fout:
	for line in tqdm(fin):
		data = json.loads(line)
		out = {}
		out['paper'] = data['paper']
		out['label'] = data['label']
		labels = data['predicted_label']
		l = len(labels)
		sim = {}
		for label, score in zip(labels, pred[i:i+l]):
			sim[label] = score
		best_parent_score = -10000
		'''
		for x in sim:
			try:
				sim[x] = 2*(sim[x] - min(sim.values())) / (max(sim.values()) - min(sim.values())) - 1
			except:
				pass
		'''	
		#for x in sim.keys():
	 #	sim[x] = (sim[x] - np.mean(list(sim.values()))) / np.std(list(sim.values()))
  
		#sim = sim - np.mean(sim.values()) / np.std(sim.values())		
		bag = set(sim.keys())
		eval_ord = []
		to_remove = set()
		while len(bag) > 0:
			for x in bag:
				flag = False
				rev = [str(a) for a in rev_dict[int(x)]]
				for r in rev:
					if r in bag:
						flag = True
						break
				if not flag:
					eval_ord.append(x)
					to_remove.add(x)
			bag -= to_remove
		# eval_ord = sim
		for lbl in eval_ord:
			#score = sim[lbl]
			for x in rev_dict[int(lbl)]:
				if str(x) in set(sim.keys()):
					best_parent_score = max(best_parent_score, sim[str(x)])
			sim[lbl] += best_parent_score if best_parent_score != -10000 else 0
		sim_sorted = sorted(sim.items(), key=lambda x:x[1], reverse=True)
		out['predicted_label'] = sim_sorted
		fout.write(json.dumps(out)+'\n')
		
		i += l
