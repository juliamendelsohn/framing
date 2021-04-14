import json 
import random
import os

def load_examples(filename):
	examples = []
	with open(filename,'r') as f:
		for line in f:
			examples.append(json.loads(line))
	return examples


def split_examples(filename):
	examples = load_examples(filename)
	random.shuffle(examples)
	split1 = examples[:int(len(examples)/2)]
	split2 = examples[int(len(examples)/2):]
	return split1,split2


def write_examples(filename,example_list):
	with open(filename,'w') as f:
		for ex in example_list:
			json.dump(ex,f)
			f.write('\n')

def main():
	base_path = '/home/juliame/potato/data/07-23-20/'
	dev_file = os.path.join(base_path,'dev.json')
	test_file = os.path.join(base_path,'test.json')
	outfile1 = os.path.join(base_path,'eval1.json')
	outfile2 = os.path.join(base_path,'eval2.json')

	dev1,dev2 = split_examples(dev_file)
	test1,test2 = split_examples(test_file)
	write_examples(outfile1,dev1+test1)
	write_examples(outfile2,dev2+test2)


if __name__ == "__main__":
	main()