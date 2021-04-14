import sys
import regex as re

FLAGS = re.MULTILINE | re.DOTALL


def remove_stopwords(tokens):
    stopwords_file = '../data/english-stopwords'
    with open(stopwords_file,'r') as f:
        stopwords = set([w.strip('\n') for w in f.readlines()])
        for t in tokens:
            if t not in stopwords:
                new_tokens.append(t)
        return new_tokens



def process_text(tweet_text,keep_hashtag=True,keep_stopwords=True,keep_possessives=True,keep_metatokens=True):
    token_string = tokenize(tweet_text,keep_hashtag) #tokenize
    if keep_possessives == False:
        token_string = re.sub(r"'s\b","",token_string) #removes possessives
    tokens = token_string.split()
    if keep_stopwords == False:
        tokens = remove_stopwords(tokens)

    if keep_metatokens == False:
        tokens = [t for t in tokens if (t[0] !='<' or t[-1] != '>')]  #removes tokens surrounded by brackets

    punctuation = "!\"$%&'()*+, -./:;=?@[\\]^_`{|}~"
    tokens = [t.strip(punctuation) for t in tokens]  #strip punctuation
    tokens = [t for t in tokens if len(t) > 0]
    return tokens


"""
Below this comment is:
Script for tokenizing tweets by Romain Paulus
with small modifications by Jeffrey Pennington
with translation to Python by Motoki Wu
Translation of Ruby script to create features for GloVe vectors for Twitter data.
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body.lower())
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"


def tokenize(text,all_lower=False,remove_hashtag=False):
    # Different regex parts for smiley faces
    #eyes = r"[8:=;]"
    #nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"@\w+", "<user>")
    # text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    # text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    # text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    # text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    # text = re_sub(r"/"," / ")
    # text = re_sub(r"<3","<heart>")
    #text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    #text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    #text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    if remove_hashtag:
        text = re_sub(r"#\S+", hashtag)
    if all_lower:
        text = text.lower()

    return text
    return text.lower()


if __name__ == '__main__':
    _, text = sys.argv
    if text == "test":
        text = "I TEST alllll kinds of #hashtags and #HASHTAGS, @mentions and 3000 (http://t.co/dkfjkdf). w/ <3 :) haha!!!!!"
    tokens = tokenize(text)
    print(tokens)