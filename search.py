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
inverted_index_path = "/Users/gordonlam/Documents/GitHub/COMP6741/Project/doc_index/inverted_index.json"
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
    return tokens


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

    #p1 = [x[0] for x in term1_posting]
    #p2 = [x[0] for x in term2_posting]
    
    # Assume term posting is list[list[int]]
    pl1 = term1_posting
    pl2 = term2_posting

    skip_p1 = int(len(pl1)**0.5)
    skip_p2 = int(len(pl2)**0.5)

    pos1,pos2 = 0,0

    #print(p1)
    #print(p2)
    #p1_skip_pointers = [i for i in range(0,len(p1),int(len(p1)**0.5))] 
    #p2_skip_pointers = [i for i in range(0,len(p2),int(len(p2)**0.5))] 
    merged_list = []
    while (pos1 < len(pl1)) and (pos2 < len(pl2)):
        # Check if the document for term 1 exist for term 2 for the first element
        if pl1[pos1] == pl2[pos2]:
            merged_list.append(pl1[pos1])
            # now we go to the next posting by removing the first term
            pos1 += 1
            pos2 += 1
        # else means no match so we can skip..., implement a skip pointer...
        else:
            if pl1[pos1] < pl2[pos2]:
                #p1 = p1[1:] # No skip logic
                next_pos1 = pos1 + skip_p1
                # check the value at position p1 if less than total AND value at P1 is 
                if next_pos1 < len(pl1) and pl1[next_pos1] <= pl2[pos2]:
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
                if next_pos2 < len(pl2) and pl2[next_pos2] <= pl1[pos1]:
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

term2_posting = [
    [1, 4],       # term appears in doc 1 at position 4
    [2, 7],       # term appears in doc 2 at position 7
    [1121, 3]     # term appears in doc 7 at positions 3 and 9
]

term1_posting = [1, 3, 7, 9, 11, 13, 21, 22, 153, 1121]
term2_posting = [1, 2, 1121]
#print([x[0] for x in term1_posting])
#print([x[0] for x in term2_posting])

posting_intersection_algorithm(term1_posting, term2_posting)
posting_union_algorithm(term1_posting, term2_posting)
#%%
a = [x[0] for x in inverted_index["five"]["postings"]]
#%%
b = [x[0] for x in inverted_index["day"]["postings"]]

#%%
posting_intersection_algorithm(a,b)
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
    
    # If all search terms must be present in the document, we start by checking if any of the terms are NOT in the index
    # if any(term not in inverted_index for term in query_terms):
    #    return []

    # Now we can proceed to find the intersection of postings lists
    """
    Theory:
    To find the intersection, we find the word term, then find the next term, until all terms are located 
    
    We use the merging algorithm provided in the lectures which is to merge two terms at a time, starting with the smallest posting list
    The posting list will then be be used to create a simplified posting list and merge successively until all the terms are searched...
    the result is a posting list of matched documents
    """

    # Reorder the query terms based on the value of the the number documents, asc
    term_df_list = [(term,inverted_index[term]["df"]) for term in set(query_terms)]
    term_df_list.sort(key=lambda x:x[1])
    term_df_list = [a for a,_ in term_df_list]

    print(f"TERM_DF: {term_df_list}")

    # In the matching docs store the posting information for the smallest doc
    matching_docs = [x[0] for x in inverted_index[term_df_list[0]]["postings"]] #only send the docID for now
    #matching_docs = inverted_index[term_df_list[0]]["postings"]

    # Remove duplicates from the matching docs first...
    matching_docs = sorted(list(set(matching_docs)),reverse=False)
    print(f"Matched docos 1st run:{matching_docs}")

    #print(matching_docs)
    # Now iterate through the list to search for docs in the II until all terms are processed
    
    # Add if incase only one term is queried
    if len(matching_docs) > 1:
        for i in range(1,len(term_df_list)):
            postings_2 = [x[0] for x in inverted_index[term_df_list[i]]["postings"]] # Creaet a simple list of documents
            #postings_2 = inverted_index[term_df_list[i]]["postings"]
            print(f"Current matching the term: {term_df_list[i]}")

            #Update matching_docs with the new posting list
            matching_docs = posting_intersection_algorithm(matching_docs,postings_2)
            print(matching_docs)
            
    return matching_docs

    # We want to return the posting list withn the posting in it

#matched_docs = match_documents("June", inverted_index)  # Use a set to avoid duplicates
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
