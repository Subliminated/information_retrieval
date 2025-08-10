import os
import re
import sys
import json
from collections import defaultdict

import nltk
#nltk.download('punkt', quiet=True)
#nltk.download('averaged_perceptron_tagger', quiet=True)
#nltk.download('wordnet', quiet=True)
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer

lemmatizer = WordNetLemmatizer()

###################################### Load functions ######################################

def load_index(index_folder, name_of_index_file):
    # Example: load a single index file (adjust as needed)
    index_file = os.path.join(index_folder, name_of_index_file)
    with open(index_file, "r", encoding="utf-8") as f:
        return json.load(f)

###################################### Preprocessing functions ######################################
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

###################################### Normalization functions ######################################
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

def stem_hyphenated(token, stemmer=None):
    """
    Stems each part of a hyphenated token and rejoins with hyphens.
    """
    if stemmer is None:
        stemmer = PorterStemmer()
    parts = token.split('-')
    stemmed_parts = [stemmer.stem(part) for part in parts]
    return '-'.join(stemmed_parts)

def handle_normalize(text):
    """
    Custom handler for processing text to remove possessives, handle abbreviations, and lemmatize.
    Inputs
    - text -> string: text string to be normalized entirely
    Outputs
    - text -> string: with possessives removed, abbreviations handled, and lemmatized
    """
    text = text.lower()
    # Handle possessives like "cat's" → "cat" by matching any alphanumeric character or underscore, ignore punctuation and spacs
    text = re.sub(r"\b(\w+)(?:'s|s')\b", r"\1", text)
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    stemmer = PorterStemmer()
    lemmatized_stemmed = []
    for token, pos in tagged_tokens:
        if '-' in token:
            lemma = lemmatize_hyphenated(token)
            stemmed = stem_hyphenated(lemma)
        else:
            lemma = lemmatizer.lemmatize(token, get_wordnet_pos(pos))
            stemmed = stemmer.stem(lemma)
        # Apply stemming after lemmatization
        lemmatized_stemmed.append(stemmed)
    # filter empty and stray quotes " ' "
    lemmatized_stemmed = [tok for tok in lemmatized_stemmed if tok and (tok != "'")]
    lemmatized_stemmed = ' '.join(lemmatized_stemmed)
    return lemmatized_stemmed


###################################### Handle numeric tokens ######################################
#%%
def normalize_numeric_tokens(text):
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    text = re.sub(r'\b\d+\.\d+\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

###################################### Full Preprocess ######################################
def preprocess_and_tokenize(text):
    text = handle_abbreviations(text) # Convert abbreviations first
    text = text.lower() # ignore case
    text = handle_hyphens(text)    # Handle hyphenated terms
    text = handle_normalize(text)    # Handle singular/plural, tense, and possessive forms through stemming/lemmatization and normalisation
    text = normalize_numeric_tokens(text)    # Handle numeric tokens, commas in tokens, and decimal places
    text = text.strip()  # Remove leading and trailing spaces      
    text = re.sub(r'\s+', ' ', text)  # Tokenize text to words, any punctuation is treated as a token divider
    tokens = re.findall(r'\b[\w-]+\b', text) #Ignore punctuations here as we dont need to tokenize
    return tokens #list[str]

###################################### Truncate Index Preprocess ######################################
def truncate_index(query_terms,inverted_index):
    return {key:inverted_index[key] for key in query_terms if key in inverted_index.keys()}

###################################### Reversed Index Preprocess ######################################

def reverse_index(inverted_index):
    """
    This function reverse an inverted index to fulfill the needs of ranked retrieval and other requirements

    Input:
    - inverted_index -> dict[dict]: inverted index of the form {'june': {4988: [[40, 21, 7],....
    
    Output:
    - reversed_index -> dict[dict]
    """
    reversed_index = {}
    for term, value in inverted_index.items():
        for docid, postings in value.items():
            # Skip non-list values (e.g., 'df' or other metadata)
            if not isinstance(postings, list):
                continue
            if docid not in reversed_index:
                reversed_index[docid] = {"terms":{}}
            if term not in reversed_index[docid]["terms"]:
                reversed_index[docid]["terms"][term] = []
            for term_tuple in postings:
                reversed_index[docid]["terms"][term].append(term_tuple)

    # initialise scores and best matching term indexes
    for docid,value in reversed_index.items():
        reversed_index[docid]["scores"] = [0,0,0,0]
        # Add best matching terms:
        """Logic for displaying lines..
        Case A: Only consider single terms , then just print occurrence
        Case B: for multiple query, simple take the term pairs and extract the lines in which they exist -> dedupe, order, print
        """
        reversed_index[docid]["best_indices"] = []
    return reversed_index
###################################### Term Coverege Score  ######################################

def calculate_coverage(query_terms,reversed_index):
    """
    Metric: = number of terms covered, easy...

    Inputs:
    - query_term -> list[str]
    - reversed_index -> dict[dict]

    Outputs:
    - reveresed_index -> dict[dict]
    """
    total_query_terms = len(query_terms)
    # Go through each document and score
    for docid,value in reversed_index.items():
        score = 0
        for term in query_terms:
            if term in value["terms"].keys():
                score+=1
        #Update scores
        reversed_index[docid]["scores"][0] = score/total_query_terms#count_keys/total_query_terms
    return reversed_index

###################################### Pair Proximity ######################################
def calculate_pair_prox(query_terms, reversed_index):
    """
    Metric: number of term (ignores decimals) between consecutive matched query terms in a document from left to right
    Ignores any unmatched terms 

    For each pair of terms ...select pair of distances that yield min prox distance. 

    Average min instance = total min distances divided by number of matched query term pairs. 

    Determine if the min distance is also an ORDERED pair i.e. left term < right term by term index position
    
    Inputs
    - query_terms -> list[str]
    - reversed_index -> dict[dict]

    Output
    - reversed_index -> dict[dict]
    """
    # First detect size of query term, the scores won't be updated
    if len(query_terms) == 0:
        return reversed_index

    if len(query_terms) == 1:
        for docid, term_dict in reversed_index.items():
            # Store pair prox score of the first entry...
            relevant_query_terms = [term for term in query_terms if term in term_dict["terms"].keys()]

            if len(relevant_query_terms) == 1:
                # Add the best first posting directly into the index.. to capture the best term sentence
                post = term_dict["terms"][relevant_query_terms.pop()][0]
                reversed_index[docid]["best_indices"].append(post)      
        return reversed_index

    # If not then compute the query term prox scores for EACH document in the reversed index
    for docid, term_dict in reversed_index.items():
        #Reduce query terms to only the keys available in the dictionary
        relevant_query_terms = [term for term in query_terms if term in term_dict["terms"].keys()]

        # Check again and break for loop if relevant query terms are less than 2. 
        if len(relevant_query_terms) <= 1:
            if len(relevant_query_terms) == 1:
                post = term_dict["terms"][relevant_query_terms.pop()][0]
                reversed_index[docid]["best_indices"].append(post)
            continue
            
        # Otherwise, calculate proximity scores
        pair_scores = []
        pair_score_list = []
        for i in range(1,len(relevant_query_terms)):
            #compute pairs then move right, sliding against the query terms in ORDER
            postings1 = term_dict["terms"][relevant_query_terms[i-1]]
            postings2 = term_dict["terms"][relevant_query_terms[i]]

            # Best score
            min_distance = 100000000 # some arbitrary large number 
            best_dist_pair = []

            # For each term e.g. june, july we compare EVERY value pair - create a lambda to find absolute difference 
            distance_lambda = lambda a,b : max(abs(a[0] - b[0])-1,0) # minus 1 since no two positions will occupy the same index in the same docuemnt

            # Go through all pairs of postings for each term
            for post1 in postings1:
                for post2 in postings2:
                    curr_dist = distance_lambda(post1,post2) #Outputs a value >= 0
                    if curr_dist == min_distance:
                        if best_dist_pair is not None:
                            # if they are equal, update ONLY if current posting is not ordered and new posting are ordered
                            if (best_dist_pair[0][0] > best_dist_pair[1][0]) and (post1[0] < post2[0]):
                                best_dist_pair = [post1,post2]
                        else:
                            #if none for whatever reason
                            best_dist_pair = [post1,post2]
                    elif curr_dist < min_distance:
                        #then update the min distance IF pair is ordered
                        best_dist_pair = [post1,post2]
                        min_distance = curr_dist 
            
            #Once you have your min_distance and best dist_pair compute best score
            term_pair_data= ({"term_pairs":[relevant_query_terms[i-1], relevant_query_terms[i]]},
                                  {"min_dist":min_distance},
                                  {"indices":best_dist_pair},
                                  {"ordered":1 if best_dist_pair[0] < best_dist_pair[1] else 0}
                                  )
            pair_scores.append(term_pair_data)

            #Also add the best pairs to the pair list,
            pair_score_list += best_dist_pair

        #Store indices for best_dict_pair in pair_scores, for each term pair however, only, these are chained...
        """
        example ab will have a indices for ab = 2 posting go through 0 times
        example abc will have will have a indices for ab and bc = 4 posting, need to go through 1 time 
        example abcd will hace an indices for ab, bc, cd = 6 posting, need to go through 2 times
        example abcde will hace an indices for ab, bc, cd, de = 8 posting, need to go through 3 times
        i.e. (len(posting) - 1) * 2 = 6 posting to do through, 
        """
        best_pair_scores = []        
        # go through all inner elements
        for i in range(1,len(relevant_query_terms)-1):
            # Check if which occurence of the term appears first
            if pair_score_list[i][0] < pair_score_list[i+1][0]:
                best_pair_scores.append(pair_score_list[i])
            else:
                best_pair_scores.append(pair_score_list[i+1])
        best_pair_scores = [pair_score_list[0]] + best_pair_scores + [pair_score_list[-1]] #Add the first and last pairscores
        # sort the key
        # best_pair_scores.sort(key=lambda x: x[1]) # No need, sorted later anyways
        reversed_index[docid]["best_indices"] += best_pair_scores

        # Once you have gone through all the pairs calculate the average prox score
        total_proximity_sum = 0
        total_ordered_pair_sum = 0
        for _, dist, _, order_score in pair_scores:
            total_proximity_sum += dist["min_dist"]
            if order_score["ordered"] == 1:
                total_ordered_pair_sum +=1
        if len(relevant_query_terms)-1 == 0:
            raise ValueError
        else:
            average_min_proximity = total_proximity_sum / (len(relevant_query_terms)-1)
            #Update the reversed index for the docID
            reversed_index[docid]["scores"][1] = 1/(1+average_min_proximity) # for prox score
            reversed_index[docid]["scores"][2] = total_ordered_pair_sum #for each ordered pair

        # Update the document scores, if we have calculated any pair scores
        if len(pair_scores) > 0:
            reversed_index[docid]["scores"][1] = 1/(1+average_min_proximity)
        else:
            reversed_index[docid]["scores"][1] = 0

    return reversed_index  # once all docIDs have been iterated through

###################################### Pair Proximity ######################################
def compute_final_score(alpha,beta,gamma,reversed_index):
    """
    Score based for a given query based on formula\
    
    Inputs
    - alpha -> float: arbitrary weight for coverage score
    - beta -> float: arbitrary weight for min proximity distance score
    - gamma -> float: arbitrary weight for ordered pair score

    Outputs
    - reversed_index -> dict[dict]
    """
    index = reversed_index
    for docid, term_dict in index.items():
        score = term_dict["scores"]
        final_score = alpha*score[0] + beta*score[1] + gamma*score[2]
        index[docid]["scores"][3] = final_score
    # Sort documents by score in descending order
    return index # with scores

def find_documents(query, inverted_index):
    """
    Match documents based on the query against the inverted index.
    The search logic uses union of postings lists for each term in the query.
    The text will first be normalized and tokenized before matching.
    Then, for each token in the query, it will find the corresponding postings list in the inverted index.
    If a term is not found, it will skip that term.

    Logic:
        We need to build a dictionary to compute 3 scores...
    Algorithm
    1) turn then: For the truncated index into an index + words e.g.
        doc 1: term_index: [word1:[index1,index2,..],word2:[index1,index2,..]] , score:[0,0,0], best_indices = []
        doc 2: term_index: [word1:[index1,index2,..],word2:[index1,index2,..]] , score:[0,0,0], best_indices = []
        doc 3: term_index: [word1:[index1,index2,..],word2:[index1,index2,..]] , score:[0,0,0], best_indices = []
    2) Coverage can be computed by len(value)/len(total query terms)
    3) Avg_pair_ditance we need to check 
        - if 1 term matched, then avg pair distance = 0 and score is 0 (e.g. if we have only 1 query term then this doesnt matter, but if we have 2 terms, and we get 1 still, it should be ranked and scored lower)
        - if 2+ terms exist, then for each pair of terms iterate through each pair of index FROM LEFT TO RIGHT, this will yield proximity scores
            - Take the minimum proximity score for each term pair e.g. if we have 4 matched terms we have 3 pairs and 3 scores: e.g. 1,3,5
                - For each Term pair score you compute, store the pair (key) and current prox (value), ONLY update key IF (1) new prox is < old prox OR if ==, one is ordered
                - Once all pairs are updated, increment the ordered pair score by 1 e.g. index[docid[score]][2] += 1
            - Average min distance sum of total dis/num of matched pairs = (1+3+5)/3 = 3
            - Reduce the prox distance to just For each min distance, compute also a ordered pair score
    4) Reduce your index to now index[docid] = compute_final_score(index[docid][score],alpha,beta,gamma)
    5) Use the final score to order your documents
    
    Inputs
    - query -> string: the search query to match against the inverted index
    
    Outputs
    - reversed_index -> dict[dict]: reversed index with updated scores and positions
    """
    query_terms = preprocess_and_tokenize(query)
    if not query_terms:
        return []  # No terms to match
    
    # Create a truncated dict of the postings (same as an OR query where every document is included)
    truncated_index = truncate_index(query_terms,inverted_index)

    # Step 1. Reverse the index
    reversed_index = reverse_index(truncated_index)

    # Step 2 Compute metrics and populate scores
    reversed_index = calculate_coverage(query_terms, reversed_index)
    
    # Step 3 Calculate Proximity Score
    reversed_index = calculate_pair_prox(query_terms, reversed_index) 

    # Step 4 Calculate Final Score
    alpha,beta,gamma = 1,1,0.1
    reversed_index = compute_final_score(alpha,beta,gamma,reversed_index)
    
    # Step 5, Provide a sorted output...
    return reversed_index



def rank_retrieve(scored_index, index_path, show_line):

    # Check if the user has displayed >'
    sorted_docs = sorted(scored_index.items(),
    key=lambda item: (-item[1]['scores'][-1], int(item[0]))
    )

    if not show_line:
        for docid, _ in sorted_docs:
            print(docid)
            pass
    else:
        #Regular print
        for docid, package in sorted_docs:
            lines = list(set([f"{line_num}" for _,line_num,_ in package["best_indices"]])) #dedupe
            lines.sort() #order
            
            file_path = os.path.join(index_path,f"{docid}.json")
            with open(file_path, "r", encoding="utf-8") as f:
                doc_json = json.load(f)

            #Now we need to open up the document and print the lines
            print(f"> {docid}")
            for line in lines:
                print(doc_json[line], end='')

if __name__ == "__main__":
    #check if two arguments are provided, always need (1) document path and (2) output_path of index files
    if len(sys.argv) != 2:
        print("Usage: python index.py <index_path>")
        print("Expected 1 arguments, got", len(sys.argv) - 1)
        sys.exit(1)
        # Expecting python index.py data doc_index
    else:
        index_path = sys.argv[1]

    #Import inverted index
    name_of_index_file = "inverted_index.json"
    inverted_index = load_index(index_path,name_of_index_file)

    while True:
        try:
            # (1) Accept a search query from the standard input
            user_input = input("")
            if user_input.startswith("> "):
                user_input = user_input[2:]
                matched_docs = find_documents(user_input, inverted_index)  # Use a set to avoid duplicates
                rank_retrieve(matched_docs,index_path,show_line = True)
            else:
                matched_docs = find_documents(user_input, inverted_index)  # Use a set to avoid duplicates
                rank_retrieve(matched_docs,index_path,show_line = False)
        except (EOFError, KeyboardInterrupt):
            break
    
# run on local: python search.py doc_index