import itertools
import os

import numpy as np

# Get all filenames of interest
filename_list = os.listdir(os.getcwd())

# Use next() + lambda + loop for Initial-Four-Characters Case Categorization
util_func = lambda x, y: x[0] == y[0] and x[1] == y[1] and x[2] == y[2] and x[3] == y[3]
res = []
for sub in filename_list:
    ele = next((x for x in res if util_func(sub, x[0])), [])
    if ele == []:
        res.append(ele)
    ele.append(sub)

# Get names of EVCoupling heterodimers
hd_list = np.loadtxt('../../evcoupling_heterodimer_list.txt', dtype=str).tolist()

# Find first heterodimer for each complex and move it to a directory categorized by the complex's PDB code
for r in res:
    if len(r) >= 2:
        r_combos = list(itertools.combinations(r, 2))[:1]  # Only select a single heterodimer from each complex
        for combo in r_combos:
            heterodimer_name = combo[0][:5] + '_' + combo[1][:5]
            if heterodimer_name in hd_list:
                pdb_code = combo[0][:4]
                os.makedirs(pdb_code, exist_ok=True)
                l_b_chain_filename = combo[0]
                l_b_chain_filename = l_b_chain_filename[:-4] + '_l_b' + '.pdb'
                r_b_chain_filename = combo[1]
                r_b_chain_filename = r_b_chain_filename[:-4] + '_r_b' + '.pdb'
                if not os.path.exists(os.path.join(pdb_code, l_b_chain_filename)):
                    os.rename(combo[0], os.path.join(pdb_code, l_b_chain_filename))
                if not os.path.exists(os.path.join(pdb_code, r_b_chain_filename)):
                    os.rename(combo[1], os.path.join(pdb_code, r_b_chain_filename))
