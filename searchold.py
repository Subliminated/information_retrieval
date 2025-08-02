#%%
import os
import re
import sys
import shutil
import json

from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

document_path = os.getcwd() + '/data'
index_path = os.getcwd() + '/doc_index'

print("docp",document_path)
print("indexp",index_path)

#%%
# Temporary import for inverted index
inverted_index_path = "/Users/gordonlam/Documents/GitHub/COMP6741/Project/doc_index/inverted_index2.json"
with open(inverted_index_path, 'r', encoding='utf-8') as f:
    inverted_index = json.load(f)

#print(inverted_index.keys())  # Print keys to verify loading
#%%

#TEST
#print([x[1] for x in inverted_index["totally"]["postings"]])
#print([x[1] for x in inverted_index["surprise"]["postings"]])

#%%
def handle_abbreviations(text):
    text = re.sub(r'\b([A-Za-z])\.', r'\1', text)
    return text
        
def handle_hyphens(text):
    def replace(match):
        token = match.group()
        parts = token.split('-')
        if len(parts) > 1 and len(parts[0]) < 3:
            return token  # preserve
        else:
            return ' '.join(parts)  # split
        
    text = text.lower()
    return re.sub(r'\b\w+(?:-\w+)+\b', replace, text)

#%%
###################################### Normalization functions ######################################

#%%
# Get wordnet POS tag and lemmatize
def get_wordnet_pos(treebank_tag):
    # Get tags for adjectives, verbs, nouns, adverbs
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun
    
def lemmatize_hyphenated(token):
    parts = token.split('-')
    tagged_parts = pos_tag(parts)
    lemmatized_parts = [
        lemmatizer.lemmatize(part, get_wordnet_pos(pos))
        for part, pos in tagged_parts
    ]
    return '-'.join(lemmatized_parts)

def handle_normalize(text):
    """
    Custom handler for processing text to remove possessives, handle abbreviations, and lemmatize.
    Inputs
    - text -> string: text string to be normalized entirely
    Outputs
    - text string with possessives removed, abbreviations handled, and lemmatized
    """
    text = text.lower()
    # Handle possessives like "cat's" → "cat" by matching any alphanumeric character or underscore, ignore punctuation and spacs
    text = re.sub(r"\b(\w+)(?:'s|s')\b", r"\1", text)
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    lemmatized = []
    for token, pos in tagged_tokens:
        if '-' in token:
            lemmatized.append(lemmatize_hyphenated(token))
        else:
            lemmatized.append(lemmatizer.lemmatize(token, get_wordnet_pos(pos)))

    lemmatized = [tok for tok in lemmatized if tok and (tok != "'")]  # filter empty and stray quotes " ' "
    lemmatized = ' '.join(lemmatized)
    return lemmatized

###################################### Handle numeric tokens ######################################
#%%
def normalize_numeric_tokens(text):
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    text = re.sub(r'\b\d+\.\d+\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

###################################### Full Preprocess ######################################

#%%
def preprocess_and_tokenize(text):
    text = handle_abbreviations(text) # Convert abbreviations first
    text = text.lower() # ignore case
    text = handle_hyphens(text)    # Handle hyphenated terms
    text = handle_normalize(text)    # Handle singular/plural, tense, and possessive forms through stemming/lemmatization and normalisation
    text = normalize_numeric_tokens(text)    # Handle numeric tokens, commas in tokens, and decimal places
    tokens = re.findall(r'\b[\w-]+\b', text) #Find tokens, any hyphenated terms are kept as is
    return tokens #list[str]


#%%
###################################### Document Matching ######################################
def posting_intersection_algorithm(term1_posting, term2_posting):
    """
    How it works...
    1) look for the terms one at a time from the first posting list...
        - For each term, check which document 
        - If not match, then pop off index and move to next comparison
    2) Implement a skip pointer mechanism that allows the pointer to move ahead and compare the document

    Inputs:
    - term1_posting -> list[list[int]]
    - term2_posting -> list[list[int]]

    Outputs:
    - merged_posting_list
    """
    # get list of postings
    #p1 = [(i,x[0]) if i%int(len(term1_posting)**0.5) == 0 else (None,x[0]) for i,x in enumerate(term1_posting)]
    #p2 = [(i,x[0]) if i%int(len(term2_posting)**0.5) == 0 else (None,x[0]) for i,x in enumerate(term2_posting)]

    # Assume term posting list[list[int]] and we want posting list to be [docid,...]
    #pl1 = [x[0] for x in term1_posting]
    #pl2 = [x[0] for x in term2_posting]
    
    # Assume term posting is list[list[int]] and we want posting list to be [[docid,indexid]]
    pl1 = term1_posting # this will be [1069,57],[1069,117],...]
    pl2 = term2_posting

    skip_p1 = int(len(pl1)**0.5)
    skip_p2 = int(len(pl2)**0.5)

    pos1,pos2 = 0,0

    last_doc_id = None

    #print(p1)
    #print(p2)
    #p1_skip_pointers = [i for i in range(0,len(p1),int(len(p1)**0.5))] 
    #p2_skip_pointers = [i for i in range(0,len(p2),int(len(p2)**0.5))] 
    merged_list = []
    # If either list has not been fully iterated through, then go ahead
    while (pos1 < len(pl1)) and (pos2 < len(pl2)):
        # Check if the document shares both terms
        if pl1[pos1][0] == pl2[pos2][0]:
            merged_list.append(pl1[pos1])
            merged_list.append(pl2[pos2])
            merged_list = sorted((merged_list), key=lambda x: (x[0], x[1]))
            
            #check the next pos, if the next pos is the same docid as current, only move that position
            next_pos1 = pos1 + 1
            next_pos2 = pos2 + 1
            if next_pos1 < len(pl1) and pl1[pos1][0] == pl1[next_pos1][0]:
                pos1 +=1
            elif next_pos2 < len(pl2) and pl2[pos2][0] == pl2[next_pos2][0]:
                pos2 +=1
            else:
                # now we go to the next posting by removing the first term for both posting
                pos1 += 1
                pos2 += 1      

                      
            #create a step to check if the next position also shares the same doc

            # BEFORE we move to the next index for BOTH list, 
            
        # else means no match so we can skip..., implement a skip pointer...
        else:
            #check if the current docID is list than docID of other list
            if pl1[pos1][0] < pl2[pos2][0]:
                #p1 = p1[1:] # No skip logic
                next_pos1 = pos1 + skip_p1
                # check the value at position p1 if less than total AND value at P1 is 
                if next_pos1 < len(pl1) and pl1[next_pos1][0] <= pl2[pos2][0]:
                    # Set the new position to skipped
                    pos1 = next_pos1
                    #print(f"POS1 SKIPPING TO {next_pos1}")
                else:
                    # Increment by 1 because the document must now be in skip range
                    pos1 += 1
            else:
                #p2 = p2[1:] # No skip logic
                next_pos2 = pos2 + skip_p2
                # check the value at position p2 if less than total AND value at P2 is 
                if next_pos2 < len(pl2) and pl2[next_pos2][0] <= pl1[pos1][0]:
                    # Set the new position to skipped
                    pos2 = next_pos2
                    #print(f"POS2 SKIPPING TO {next_pos2}")
                else:
                    # Increment by 1 because the document must now be in skip range
                    pos2 += 1
    #return sorted(list(set(merged_list)),reverse=False)
    return merged_list
#%%
def posting_union_algorithm(term1_posting, term2_posting):
    """
    Implements a union (OR query) between two posting lists with optional skip pointers.
    
    Inputs:
    - term1_posting -> list[int] : sorted list of docIDs
    - term2_posting -> list[int] : sorted list of docIDs

    Output:
    - merged_posting_list : list[int] with all unique docIDs from either list, sorted
    """

    pl1 = term1_posting 
    pl2 = term2_posting
    union_list = sorted(set(pl1 + pl2))
    return union_list
#%%
# Examples

# e.g. APPLES
term1_posting = [
    [1, 5], # term appears in doc 1 at positions 5,7,123
    [1, 7],
    [1, 123],    
    [3, 2],       # term appears in doc 3 at position 2
    [7, 8],
    [9, 8],
    [11, 8],
    [13, 8],
    [21, 8],
    [22, 8],
    [153, 8],
    [1121, 8],
        # term appears in doc 7 at positions 8 and 15
]

# e.g. PEARS
term2_posting = [
    [1, 4],       # term appears in doc 1 at position 4
    [2, 7],       # term appears in doc 2 at position 7
    [1121, 3]     # term appears in doc 7 at positions 3 and 9
]
posting_intersection_algorithm(term1_posting, term2_posting)


#%%
term1_posting = [1, 3, 7, 9, 11, 13, 21, 22, 153, 1121]
term2_posting = [1, 2, 1121]
#print([x[0] for x in term1_posting])
#print([x[0] for x in term2_posting])

posting_union_algorithm(term1_posting, term2_posting)
#%%
a = [x[0] for x in inverted_index["five"]["postings"]]
#%%
b = [x[0] for x in inverted_index["day"]["postings"]]

#%%
posting_intersection_algorithm(a,b)

#%%
#extract unique documents
def get_unique_document_list(inverted_index):
    doc_set = set()
    for k,_ in inverted_index.items():
        doc_set = doc_set.union(x[0] for x in inverted_index[k]["postings"])
    ordered_doc_list = sorted(list(doc_set),reverse=False)
    return ordered_doc_list

inverted_index = {
    "068": {
        "df": 1,
        "postings": [
            [3750, 65],
            [3750, 68]
        ]
    },
    "08": {
        "df": 1,
        "postings": [
            [3111, 128]
        ]
    },
    "0830": {
        "df": 1,
        "postings": [
            [3463, 82]
        ]
    },
    "0900": {
        "df": 2,
        "postings": [
            [2777, 137]
        ]
    }
}
get_unique_document_list(inverted_index)
#%%
def match_documents(query, inverted_index):
    """
    Match documents based on the query against the inverted index.
    The search logic uses intersection of postings lists for each term in the query.
    The text will first be normalized and tokenized before matching.
    Then, for each token in the query, it will find the corresponding postings list in the inverted index.
    If a term is not found, it will skip that term.
    
    Inputs
    - query -> string: the search query to match against the inverted index
    
    Outputs
    - matched_docs -> list: a list of document IDs that match the query
    """
    query_terms = preprocess_and_tokenize(query)
    if not query_terms:
        return []  # No terms to match
    print(f"query terms: {query_terms}")
    
    # If all search terms must be present in the document, we start by checking if any of the terms are NOT in the index
    # if any(term not in inverted_index for term in query_terms):
    #    return []

    # Create a truncated list of the postings (same as an OR query where every document is included)
    truncated_index = {key:inverted_index[key] for key in query_terms if key in inverted_index.keys()}
    print(f"query index: {truncated_index}")

    # We can now use this to apply our 3 ranking mechanisms
    """
    we now need to build a dictionary to compute 3 scores...
    Algorithm
    1) turn then: For the truncated index into an index + words e.g.
        doc 1: term_index: [word1:[index1,index2,..],word2:[index1,index2,..]] , score:[0,0,0]
        doc 2: term_index: [word1:[index1,index2,..],word2:[index1,index2,..]] , score:[0,0,0]
        doc 3: term_index: [word1:[index1,index2,..],word2:[index1,index2,..]] , score:[0,0,0]
    2) Coverage can be computed by len(value)/len(total query terms) EZEZEZ
    3) Avg_pair_ditance we need to check 
        - if 1 term matched, then avg pair distance = 0 and score is 0 (e.g. if we have only 1 query term then this doesnt matter, but if we have 2 terms, and we get 1 still, it should be ranked and scored lower)
        - if 2+ terms exist, then for each pair of terms iterate through each pair of index FROM LEFT TO RIGHT, this will yield proximity scores
            - Take the minimum proximity score for each term pair e.g. if we have 4 matched terms we have 3 pairs and 3 scores: e.g. 1,3,5
                - For each Term pair score you compute, store the pair (key) and current prox (value), ONLY update key IF (1) new prox is < old prox OR if ==, one is ordered
                - Once all pairs are updated, increment the ordered pair score by 1 e.g. index[docid[score]][2] += 1
            - Average min distance sum of total dis/num of matched pairs = (1+3+5)/3 = 3
            - Reduce the prox distance to just For each min distance, compute also a ordered pair score
    4) Reduce your index to now index[docid] = final_score(ndex[docid][score],alpha,beta,gamma)
    5) Use the final score to order your documents
    
    DONE!

    


    For each document.... Compute...
    Coverage (easy... for each query term -> go through posting list for the term -> against each docID, add a value (e.g. 0) and increment -> after all terms, divide by # of terms
    Proximity 
    Order_score
    Total_score
    """
    #list = rank_documents(query_terms, truncated_index)


    # Reorder the query terms based on the value of the the number documents, asc
    term_df_list = [(term,inverted_index[term]["df"]) for term in set(query_terms)]
    term_df_list.sort(key=lambda x:x[1])
    term_df_list = [a for a,_ in term_df_list]

    #print(f"TERM_DF: {term_df_list}")
    #doc_ids = get_unique_document_list(query_terms)
    
    # New logic for union
    #return doc_ids
    #return matching_docs

    # We want to return the posting list withn the posting in it

matched_docs = match_documents("June", inverted_index)  # Use a set to avoid duplicates
#print(f"Matched documents: {sorted(matched_docs)}")
#%%
###################################### Document ranking ######################################

def rank_documents(doc_list, document_scores):
    """
    Rank documents based on their scores for a given query.
    """
    # Sort documents by score in descending order
    ranked_docs = sorted(document_scores.items(), key=lambda item: item[1], reverse=True)
    return ranked_docs

def rank_sorting(query, inverted_index):
    """
    From the list of documents generated, we now need to score each
    """
    #score1 = score_1()
    #score2 = score_2()
    #score3 = score_3()
    alpha = 0.5
    beta = 0.3
    gamma = 0.2

    #total_score = alpha * (matched_query_terms/total_query_terms) + beta * 1/(1+avg_pair_distance) + gamma * ordered_pairs
    

    # Sorted list
    # sort order is by descending order of score and then document IDs
    scorelist = []

    pass


def return_results(query, inverted_index):
    """
    Given a query and an inverted index, return the ranked results.
    """
    # Split the query into terms
    terms = query.lower().split()
    
    # Initialize a dictionary to hold the scores for each document
    scores = {}
    
    # Iterate through each term in the query
    for term in terms:
        if term in inverted_index:
            postings = inverted_index[term]
            for doc_id, positions in postings.items():
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += len(positions)  # Increment score by number of occurrences
    
    # Sort documents by score in descending order
    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_docs

if __name__ == "__main__":
    #check if two arguments are provided, always need (1) document path and (2) output_path of index files
    if len(sys.argv) != 3:
        print("Usage: python index.py <document_path> <index_path>")
        print("Expected 2 arguments, got", len(sys.argv) - 1)
        sys.exit(1)
        # Expecting python index.py data doc_index
    else:
        document_path = sys.argv[1]
        output_path = sys.argv[2]

    while 1==1:
        try:
            # (1) Accept a search query from the standard input
            user_input = input("")
            matched_docs = match_documents(user_input, inverted_index)  # Use a set to avoid duplicates
            # (2) Output the result to the standard output as a sequence of document names (same as their document IDs)
            for i in matched_docs:
                print(f"{i}")
        except (EOFError, KeyboardInterrupt):
            print("Exiting search.")
            break
    
    #try June,july
