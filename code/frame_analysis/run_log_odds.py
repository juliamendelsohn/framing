import os
import sys
import glob
from itertools import combinations



#base_path = "/shared-1/projects/framing/prelim-data/"
wc_path = "/shared/2/projects/framing/intermediate_results/word_counts/"
logodds_path = "/shared/2/projects/framing/intermediate_results/log_odds_08-03/"

if not os.path.exists(logodds_path):
	os.mkdir(logodds_path)

# first_file = os.path.join(wc_path,"EU.json")
# second_file = os.path.join(wc_path,"GB.json")
# prior_file = os.path.join(wc_path,"full.json")
# out_file = os.path.join(logodds_path,"EU_vs_GB.tsv")
# program = "python log_odds.py" + " -f " + first_file + " -s " + second_file + " -p " + prior_file + " --out_file " + out_file + " --min_count=5"
# os.system(program)

# first_file = os.path.join(wc_path,"EU.json")
# second_file = os.path.join(wc_path,"US.json")
# prior_file = os.path.join(wc_path,"full.json")
# out_file = os.path.join(logodds_path,"EU_vs_US.tsv")
# program = "python log_odds.py" + " -f " + first_file + " -s " + second_file + " -p " + prior_file + " --out_file " + out_file + " --min_count=5"
# os.system(program)


# first_file = os.path.join(wc_path,"GB.json")
# second_file = os.path.join(wc_path,"US.json")
# prior_file = os.path.join(wc_path,"full.json")
# out_file = os.path.join(logodds_path,"GB_vs_US.tsv")
# program = "python log_odds.py" + " -f " + first_file + " -s " + second_file + " -p " + prior_file + " --out_file " + out_file + " --min_count=5"
# os.system(program)


country_pairs = combinations(['EU','GB','US'],2)
for (c1,c2) in country_pairs:
	for ngram_type in ['lex_frame_bigrams','unigrams','bigrams']:
		first_file = os.path.join(wc_path,'by_country',c1, ngram_type + '.json')
		second_file = os.path.join(wc_path,'by_country',c2, ngram_type + '.json')
		prior_file = os.path.join(wc_path,'by_country','full_counts', ngram_type + '.json')
		out_file = os.path.join(logodds_path, c1 + '_vs_' + c2 + '_' + ngram_type + '.tsv')
		program = "python log_odds.py" + " -f " + first_file + " -s " + second_file + " -p " + prior_file + " --out_file " + out_file + " --min_count=5"
		os.system(program)

for setting in ['norm','raw']:
	for ngram_type in ['lex_frame_bigrams','unigrams','bigrams']:
		first_file = os.path.join(wc_path,'by_ideology',setting, 'conservative', ngram_type + '.json')
		second_file = os.path.join(wc_path,'by_ideology',setting, 'liberal', ngram_type + '.json')
		prior_file = os.path.join(wc_path,'by_country','US', ngram_type + '.json')
		out_file = os.path.join(logodds_path, 'conservative_vs_liberal_' + setting + '_' + ngram_type + '.tsv')
		program = "python log_odds.py" + " -f " + first_file + " -s " + second_file + " -p " + prior_file + " --out_file " + out_file + " --min_count=5"
		os.system(program)

