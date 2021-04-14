import os 
import ast
import json
from collections import defaultdict
from itertools import combinations

def load_field_lexicon(lexicon_file):
	lexicon = defaultdict(list)
	with open(lexicon_file,'r') as f:
		for line in f:
			frame = line.split('[')[0]
			words = ast.literal_eval('[' + line.split('[')[1])
			lexicon[frame] += words
	return lexicon


def get_mfc_text(corpus_file,codes_file,overlap='intersect'):
	corpus_text = defaultdict(list)
	with open(corpus_file,'r') as f:
		corpus = json.load(f)
	with open(codes_file,'r') as f :
		codes = json.load(f)
	for article in corpus:
		text = corpus[article]['text']
		annotations = corpus[article]['annotations']['framing']
		frame_spans = []
		for annotator in annotations:
			for label in annotations[annotator]:
				start_char = label['start']
				end_char = label['end']
				frame = codes[str(label['code'])]
				frame_span = (frame,start_char,end_char)
				frame_spans.append(frame_span)
		for frame in codes.values():
			spans = [x[1:] for x in frame_spans if x[0]==frame]
			if len(spans) > 0:
				non_overlap_spans = handle_overlapping_spans(spans,overlap=overlap)
				print(frame,spans)
				print(frame,non_overlap_spans)
				for span in non_overlap_spans:
					start_char = span[0]
					end_char = span[1]
					corpus_text[frame].append(text[start_char:end_char])
	return corpus_text


def write_mfc_corpus_text(corpus_text_dict,outfile):
	with open(outfile,'w') as f:
		json.dump(corpus_text_dict,f)


def handle_overlapping_spans(spans,overlap='intersect'): #overlap is either intersect or union
	ranges = list(range(s[0],s[1]+1) for s in spans)
	non_overlap_spans = [(min(r),max(r)) for r in ranges]
	for (x,y) in combinations(ranges,2):
		intersect = set(x).intersection(set(y))
		if len(intersect) > 0:
			x_minmax = (min(x),max(x))
			y_minmax = (min(y),max(y))
			if x_minmax in non_overlap_spans:
				non_overlap_spans.remove(x_minmax)
			if y_minmax in non_overlap_spans:
				non_overlap_spans.remove(y_minmax)
			if overlap == 'intersect':
				non_overlap_spans.append((min(intersect),max(intersect)))
			elif overlap == 'union':
				union = set(x).union(set(y))
				non_overlap_spans.append((min(union),max(union)))
	return non_overlap_spans


			


def main(): 
	field_lexicon_file = "/Users/juliame/framing/data/field_lexicon.txt"
	field_lexicon = load_field_lexicon(field_lexicon_file)
	mfc_file = "/Users/juliame/framing/data/card_corpus/immigration/immigration_labeled.json"
	mfc_codes_file = "/Users/juliame/framing/data/card_corpus/codes.json"
	mfc_outfile_intersect = "/Users/juliame/framing/data/mfc_text_intersect.json"
	mfc_outfile_union = "/Users/juliame/framing/data/mfc_text_union.json"
	corpus_text_intersect = get_mfc_text(mfc_file,mfc_codes_file,overlap='intersect')
	corpus_text_union = get_mfc_text(mfc_file,mfc_codes_file,overlap='union')
	write_mfc_corpus_text(corpus_text_intersect,mfc_outfile_intersect)
	write_mfc_corpus_text(corpus_text_union,mfc_outfile_union)







if __name__ == "__main__":
	main()