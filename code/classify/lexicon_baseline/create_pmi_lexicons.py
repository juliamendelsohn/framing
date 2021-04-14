from preprocess_text import process_text 
import json 
from collections import defaultdict,Counter
from math import log


#treat each frame as doc?
def count_mfc_text(mfc_file):
	counts = defaultdict(lambda: Counter())
	with open(mfc_file,'r') as f:
		mfc_text = json.load(f)
	frames = [x for x in mfc_text.keys() if x.split()[-1] not in ['headline','primary','primany']]
	for frame in frames:
		for span in mfc_text[frame]:
			tokens = process_text(span)
			for token in tokens:
				counts[frame][token] += 1
		print(frame,counts[frame].most_common(10))
	return counts

def calculate_pmi(word_counts,threshold=5):
	pmi_scores = defaultdict(list)
	all_words_full = sum([sum(word_counts[f].values()) for f in word_counts])
	for frame in word_counts.keys():
		unsorted_scores = []
		all_words_in_frame = sum([word_counts[frame][w] for w in word_counts[frame]])
		for word in word_counts[frame]:
			w_in_frame = word_counts[frame][word]
			w_full = sum([word_counts[f][word] for f in word_counts])
			if w_full > threshold:
				p_w_given_d = w_in_frame / all_words_in_frame
				p_w = w_full / all_words_full
				try:
					pmi = log(p_w_given_d/p_w)
				except:
					pmi = 0
				unsorted_scores.append((word,pmi))
		pmi_scores[frame] = sorted(unsorted_scores,key=lambda x: x[1],reverse=True)
	return pmi_scores

def write_pmi(pmi_scores,outfile):
	with open(outfile,'w') as f:
		json.dump(pmi_scores,f)


def main():
	threshold = 50
	mfc_file = "/Users/juliame/framing/data/mfc_text_intersect.json"
	pmi_outfile = f"/Users/juliame/framing/data/pmi_mfc_intersect_thresh_{threshold}.json"
	word_counts = count_mfc_text(mfc_file)
	pmi_scores = calculate_pmi(word_counts,threshold=threshold)
	write_pmi(pmi_scores,pmi_outfile)

if __name__ == "__main__":
	main()