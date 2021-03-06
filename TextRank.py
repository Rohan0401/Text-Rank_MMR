import io
import nltk
import itertools
from operator import itemgetter
import networkx as nx
import os
import glob
import re

#apply syntactic filters based on POS tags
def filter_for_tags(tagged, tags=['NN', 'JJ', 'NNP']):
    return [item for item in tagged if item[1] in tags]

def normalize(tagged):
    return [(item[0].replace('.', ''), item[1]) for item in tagged]

def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in itertools.ifilterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

def lDistance(firstString, secondString):
    "Function to find the Levenshtein distance between two words/sentences - gotten from http://rosettacode.org/wiki/Levenshtein_distance#Python"
    if len(firstString) > len(secondString):
        firstString, secondString = secondString, firstString
    distances = range(len(firstString) + 1)
    for index2, char2 in enumerate(secondString):
        newDistances = [index2 + 1]
        for index1, char1 in enumerate(firstString):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1])))
        distances = newDistances
    return distances[-1]

def buildGraph(nodes):
    "nodes - list of hashables that represents the nodes of the graph"
    gr = nx.Graph() #initialize an undirected graph
    gr.add_nodes_from(nodes)
    nodePairs = list(itertools.combinations(nodes, 2))

    #add edges to the graph (weighted by Levenshtein distance)
    for pair in nodePairs:
        firstString = pair[0]
        secondString = pair[1]
        levDistance = lDistance(firstString, secondString)
        gr.add_edge(firstString, secondString, weight=levDistance)

    return gr



def extractSentences(text):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentenceTokens = sent_detector.tokenize(text.strip())
    graph = buildGraph(sentenceTokens)

    calculated_page_rank = nx.pagerank(graph, weight='weight')

    #most important sentences in ascending order of importance
    sentences = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)

    #return a 100 word summary
    summary = ' '.join(sentences)
    summaryWords = summary.split()
    summaryWords = summaryWords[0:101]
    summary = ' '.join(summaryWords)

    return summary

def writeFiles(summary, fileName):
    
  

    print "Generating output to " + '/home/rohan/Downloads/Final_Project/System_Summaries/TexRank/' + fileName
    summaryFile = io.open('/home/rohan/Downloads/Final_Project/System_Summaries/TexRank/' + fileName, 'w')
    summaryFile.write(summary)
    summaryFile.close()

    print "-"


#retrieve each of the articles
fil2 = sorted(glob.glob('/home/rohan/Downloads/Final_Project/Documents1/d*t'))

articles = sorted(os.listdir("/home/rohan/Downloads/Final_Project/Document1"))
for article in articles:
        print 'Reading file/' + article
        articleFile = io.open('/home/rohan/Downloads/Final_Project/Document1/'+ article, 'r')
        text = articleFile.read()
        text = re.sub(r'\w*\d\w*','', text).strip()
        text = re.sub('<DOC>','',text).strip()
        text = re.sub('<DOCNO>','',text).strip()
        text = re.sub('<DOCTYPE>','',text).strip()
        text = re.sub('<DOC>','',text).strip()
        text = re.sub('\n','',text).strip()
        text = re.sub('</DOCNO>','',text).strip()
        text = re.sub('</DOCTYPE>','',text).strip()
        text = re.sub('</DOC>','',text).strip()
        text = re.sub('<TXTTYPE>','',text).strip()
        text = re.sub('</TXTTYPE>','',text).strip()
        text = re.sub('<TEXT>','',text).strip()
        text = re.sub('</TEXT>','',text).strip()
        text = re.sub('NEWS','',text).strip()
        text = re.sub('NEWSWIRE','',text).strip()
        summary = extractSentences(text)
        writeFiles(summary, article)