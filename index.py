#%% # Import libraries
import os
import re
import sys
import shutil
import json

# Add a sorting algorithm to the inverted index
import bisect
from collections import defaultdict

#%% # Add nltk and lemmatization function
import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer


lemmatizer = WordNetLemmatizer()
#%% # Document processing function
document_path = os.getcwd() + '/Project/data/'
index_path = os.getcwd() + '/Project/doc_index/'
###################################### Handle capitalisation and abbreviation ######################################
#%% 
def handle_abbreviations(text):
    # Find all tokens in the text that is an abbreviation, then remove the full stop, keep the capitalisation
    text = re.sub(r'\b([A-Za-z])\.', r'\1', text)
    return text


###################################### Handle Hyphens ######################################

#%%
def replace(match):
        token = match.group()
        parts = token.split('-')
        if len(parts) > 1 and len(parts[0]) < 3:
            return token  # preserve
        else:
            return ' '.join(parts)
        
def handle_hyphens(text):
    """
    Custom handler for processing text with hyphenated terms.
    Inputs
    - text -> string: text string to be normalized entirely
    Outputs
    - text string with hyphenated terms replaced
    """
    text = text.lower()
    return re.sub(r'\b\w+(?:-\w+)+\b', replace, text)

#s = "D-Kans co-author in-depth set-aside five-year"
#s = "The cat's ex-wives and cats' toys were playing."

#handle_hyphens(s)  # Example usage

#%%
###################################### Normalization functions ######################################
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
#get_wordnet_pos('v')  # Example usage
#%%
def lemmatize_hyphenated(token):
    parts = token.split('-')
    # POS tag each part individually
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

def normalize_text(text):
    """
    Custom handler for processing text to remove possessives, handle abbreviations, and lemmatize+stem.
    Inputs
    - text -> string: text string to be normalized entirely
    Outputs
    - text string with possessives removed, abbreviations handled, lemmatized, and stemmed
    """
    text = text.lower()
    # Handle possessives like "cat's" → "cat" by matching any alphanumeric character or underscore, ignore punctuation and spaces
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


#s = "The The US. u.S. US cat's ex-wives and cats' toys were playing."
#s = handle_hyphens(s)
#normalize_text(s)
###################################### Sentence Index ######################################

#%%
def split_into_sentences(text):
    # Rule: Only split on '.', '?', or '!' followed by space or end of text
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    print(sentence_endings)
    sentences = sentence_endings.split(text.strip())
    return sentences

#normalized = normalize_text("This is a sentence. Is it? Yes! Let's see if it works. It should work well.")
#split_into_sentences(normalized)

###################################### Handle numeric tokens ######################################
#%%
# Numbers with decimal places ignored from index, keep the integer
def normalize_numeric_tokens(text):
    """
    - Removes commas from numeric tokens (e.g., 1,000,000 -> 1000000)
    - Removes numbers with decimal places (e.g., 3.14, 1,000.50)
    - Keeps integers (e.g., 2023, 1000000)
    """
    # 1. Remove commas in numbers (e.g., 1,000,000 -> 1000000)
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    # 2. Remove numbers with decimal places (e.g., 3.14, 1,000.50)
    text = re.sub(r'\b\d+\.\d+\b', '', text)
    # 3. (Optional) Remove extra spaces left by removals
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Example usage:
#s = "The US. u.S. US population is 1,000,000. The year is 2023. Pi is 3.14. The price is 1,000.50."
#normalize_numeric_tokens(s)

###################################### Full Preprocess ######################################

#%%
def preprocess_and_tokenize(text):
    """
    Tokenize the input text into words.
    This function uses regex to find all words in the text and convert them to lowercase.

    The key requiremnents are to enable a search engine that can handle all matching rules.

    General tokenization rules:
    - Search is case insensitive.
    - Full stops for abbreviations are ignored, e.g., U.S., US are the same.
    - Hyphenated terms are split unless the first part is fewer than 3 letters, in which case the full term is preserved, e.g., D-Kans, co-author, in-depth are preserved, while set-aside, five-year are split.
    - Singular/Plural is ignored. e.g., cat, cats, cat's, cats' are all the same, ex-wives and ex-wife are the same.
    - Tense is ignored. e.g., breaches, breach, breached, breaching are all the same, co-author and co-authored
    - are the same.
    
    Sentence tokenization is handled as follows:
    - A sentence can only end with a full stop, a question mark, or an exclamation mark.
    
    Numeric tokens are handled as follows:
    - Numbers with decimal places can be ignored from the index, if you wish, as a decimal number is not a valid search term (since '.' is not allowed).
    - Numeric tokens such as years, integers should be indexed accordingly and searchable.
    - Commas in numeric tokens are ignored, e.g., 1,000,000 and 1000000 are the same.
    
    Except the above, all other punctuation should be treated as token dividers.
    """
    # Abbreviations applied first and converted to 

    text = handle_abbreviations(text)  # Handle abbreviations like U.S. -> US

    # Convert text to lowercase, since search is case insensitive. Will impact abbreviations
    # and hyphenated terms, so this should be done after handling abbreviations.
    text = text.lower() 

    # Handle hyphenated terms
    text = handle_hyphens(text)  # Process hyphenated terms
    
    # Handle singular/plural, tense, and possessive forms through stemming/lemmatization and normalisation
    text = normalize_text(text)

    # Handle numeric tokens, commas in tokens, and decimal places
    text = normalize_numeric_tokens(text)
    text = text.strip()  # Remove leading and trailing spaces      
  
    # Tokenize text to words, any punctuation is treated as a token divider
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    
    # \b[\w-]+\b matches words that may include hyphens
    #tokens = re.findall(r'\b[\w-]+\b', text) # Remove punctuations
    tokens = re.findall(r'\b[\w-]+\b|[.!?]', text) # keep punctuations for now
    return tokens
    #return text

# string with lots of punctuation and special characters
#preprocess_and_tokenize(s)
#%%
def handle_paths(document_path, index_path):
    """
    Handle the paths for document and index.
    If the document path is a file, continue, otherwise throw error
    If the index path 

    Inputs:
    - document_path: str, path to the document or directory containing documents
    - index_path: str, path to the index file or directory (optional)
    """
    if not os.path.exists(document_path):
        raise FileNotFoundError(f" path '{document_path}' does not exist.")

    # If index_path already exists, we will remove all content in the directory
    if os.path.exists(index_path):
        #Remove all contents and subdirectories of index_path
        for filename in os.listdir(index_path):
            file_path = os.path.join(index_path, filename)
            # if the path is a file or a symbolic link, remove it
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            # if the path is a directory, remove it recursively
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    # recreate index_path regardless if the directory exists
    os.makedirs(index_path, exist_ok=True)

#document_path = os.getcwd() + '/Project/data/'
#index_path = os.getcwd() + '/Project/doc_index/'
#print(os.getcwd())
#handle_paths(document_path, index_path)
#%%
###
# Sort the inverted index by docid and position during insertion
def insert_sorted(list, item):
    """
    list: postings= current list
    item: new_posting = new thing to add
    
    """

    # Binary search to find the correct insertion point
    index = bisect.bisect_left(list, item)
    list.insert(index, item)

###
#%%
def create_index(document_path, index_path):
    """
    Index the documents in the specified folder.
    """
    # Check if the document and index paths are valid, if yes, then recreate index in index_path
    handle_paths(document_path, index_path)

    # Now process the documents into an inverted index
    files_to_process = [
        os.path.join(document_path, f) for f in os.listdir(document_path)
        if os.path.isfile(os.path.join(document_path, f))
    ]    
    # Form the inverted index - create a list object to have an empty list for each inverted index term
    #inverted_index = defaultdict(list)
    inverted_index = defaultdict(lambda: {})

    #inverted_index = defaultdict(tuple) 
    total_tokens = 0

    # Go through each file in the folder
    for filename in files_to_process:
        # First reach the file and for each word, create a posting list with the document ID that contains the word. 
        with open(filename, 'r', encoding='utf-8') as file:
            docid = int(os.path.basename(filename))  # Get the file name without the path
            
            # ALTERNATIVELY Process each line individually in the document to store the line and the sentence all at once!
            lines = file.readlines()
            
            #In your index, create a json file for each document where the key is the line number and the value is the string
            indexed_docid_path = os.path.join(index_path, f"{docid}.json")

            doc_index = {index:value for index,value in enumerate(lines)}
            with open(indexed_docid_path, 'w', encoding='utf-8') as file:
                #json.dump(sorted_index, file, ensure_ascii=False, indent=None,)
                json.dump(doc_index, file, ensure_ascii=False, indent=None, separators=(',', ':'))
            
            #Now Create an inverted index
            sentence_pos=0
            #line_pos=0
            term_pos=0

            # Note the term tag will be a tuple of (sentence_pos, line_pos, term_pos)
            for line_pos, line in enumerate(lines):
                #this yields tokens
                tokens = preprocess_and_tokenize(line)
                for term in tokens:
                    # If the term is not already in the dictionary, then initialise
                    if term in ("!",".","?"):
                        sentence_pos +=1
                    else:
                        if term not in inverted_index:
                            inverted_index[term] = {docid: [(term_pos, line_pos, sentence_pos)]}
                            term_pos +=1
                        else:
                            # Do a bisected insert --NOT needed
                            #insert_sorted(inverted_index[term][docid], (term_pos, line_pos, sentence_pos))
                            if docid not in inverted_index[term]:
                                inverted_index[term][docid] = []
                            inverted_index[term][docid].append((term_pos, line_pos, sentence_pos))
                        total_tokens += 1

    # Save the inverted index to the output path as a text file
    output_path = os.path.join(index_path, 'inverted_index.json')

    # Sort the dictionary by the term before dumping to JSON
    sorted_index = {word: inverted_index[word] for word in sorted(inverted_index.keys())}
    with open(output_path, 'w', encoding='utf-8') as file:
        #json.dump(sorted_index, file, ensure_ascii=False, indent=None, separators=(',', ':'))
        json.dump(sorted_index, file, ensure_ascii=False, indent=None)

    # Finally, print the number of documents, tokens, and terms in the index
    n_doc = len(files_to_process)  # Number of documents is the number of files processed
    n_token = total_tokens #sum(len(postings) for postings in inverted_index.values())  # Total
    n_terms = len(inverted_index)  # Number of unique terms in the index
    print(f"Total number of documents: {n_doc}")
    print(f"Total number of tokens: {n_token}")
    print(f"Total number of terms: {n_terms}")
 
#create_index(document_path, index_path)
#%%

if __name__ == "__main__":
    #check if two arguments are provided, always need (1) document path and (2) output_path of index files
    if len(sys.argv) != 3:
        print("Usage: python index.py <document_path> <index_path>")
        print("Expected 2 arguments, got", len(sys.argv) - 1)
        sys.exit(1)
    else:
        document_path = sys.argv[1]
        output_path = sys.argv[2]

    create_index(document_path, output_path)

# run on local: python index.py ./data ./doc_index
# run on CSE: python3 index.py /home/cs6714/Public/data doc_index