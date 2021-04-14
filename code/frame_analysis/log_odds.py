#!/usr/bin/env python3
from collections import defaultdict
import math
import argparse
import json
import sys


### Each input file (first, second, prior) is a dict with words as keys and counts as values

parser = argparse.ArgumentParser(description='Computes the weighted log-odds-ratio, informative dirichlet prior algorithm')
parser.add_argument('-f','--first', help='Description for first counts file ')
parser.add_argument('-s','--second', help='Description for second counts file')
parser.add_argument('-p','--prior', help='Description for prior counts file')
parser.add_argument('--out_file', type=argparse.FileType('w'), default=sys.stdout)
parser.add_argument('--min_count', type=int,default=0)
parser.add_argument('--stopwords')
args = parser.parse_args()

def LoadCounts(filename, min_count=0, stopwords=set()):
  result = defaultdict(int)
  word_counts = json.load(open(filename))
  for word, count in word_counts.items():
    if count >= min_count and word not in stopwords:
      result[word] = count
  return result
  
def LoadStopwords(filename):
  stopwords = set()
  for line in open(filename):
    for word in line.split():
      if word:
        stopwords.add(word)
  return stopwords

def ComputeLogOdds(counts1, counts2, prior):
  sigmasquared = defaultdict(float)
  sigma = defaultdict(float)
  delta = defaultdict(float)

  for word in prior.keys():
      prior[word] = int(prior[word] + 0.5)

  for word in counts2.keys():
      counts1[word] = int(counts1[word] + 0.5)
      if prior[word] == 0:
          prior[word] = 1

  for word in counts1.keys():
      counts2[word] = int(counts2[word] + 0.5)
      if prior[word] == 0:
          prior[word] = 1

  n1  = sum(counts1.values())
  n2  = sum(counts2.values())
  nprior = sum(prior.values())


  for word in prior.keys():
      if prior[word] > 0:
          l1 = float(counts1[word] + prior[word]) / (( n1 + nprior ) - (counts1[word] + prior[word]))
          l2 = float(counts2[word] + prior[word]) / (( n2 + nprior ) - (counts2[word] + prior[word]))
          sigmasquared[word] =  1/(float(counts1[word]) + float(prior[word])) + 1/(float(counts2[word]) + float(prior[word]))
          sigma[word] =  math.sqrt(sigmasquared[word])
          delta[word] = ( math.log(l1) - math.log(l2) ) / sigma[word]

  return delta

def main():
  stopwords = set()
  if args.stopwords:
    stopwords = LoadStopwords(args.stopwords)
  #else:
  #  print("Not using stopwords")

  counts1 = LoadCounts(args.first, args.min_count, stopwords)
  counts2 = LoadCounts(args.second, args.min_count, stopwords)
  prior = LoadCounts(args.prior, args.min_count, stopwords)

  delta = ComputeLogOdds(counts1, counts2, prior)
  for word, log_odds in sorted(delta.items(), key=lambda x: x[1]):
    #print word.encode('utf-8'),log_odds
    args.out_file.write("{}\t{:.3f}\n".format(word, log_odds))

if __name__ == '__main__':
  main()