import os
import sys
import subprocess

#%%
#import os
import subprocess

#test_dir = "/Users/gordonlam/Documents/GitHub/COMP6741/Project/test"
#index_py_path = "/Users/gordonlam/Documents/GitHub/COMP6741/Project/index.py"
#search_py_path = "/Users/gordonlam/Documents/GitHub/COMP6741/Project/search.py"
#index_path = "/Users/gordonlam/Documents/GitHub/COMP6741/Project/doc_index"
#query = "Apple"
#result = subprocess.run([
#    'python3', search_py_path, index_path
#], input=query, text=True, capture_output=True)
#result

#%%
print("Current working directory:", os.getcwd())
print("Current file directory:", os.path.dirname(os.path.abspath(__file__)))

#%%
os.getcwd()
#%%
def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def parse_output(output):
    # Split output into lines, remove empty lines, and strip whitespace
    return [line.strip() for line in output.strip().split('\n') if line.strip()]

def compute_metrics(expected, actual):
    expected_set = set(expected)
    actual_set = set(actual)
    tp = len(expected_set & actual_set)
    fp = len(actual_set - expected_set)
    fn = len(expected_set - actual_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, fscore

def run_test(query_file, answer_file,index_py_path, search_py_path, index_path):
    query = read_file(query_file)
    expected_output = parse_output(read_file(answer_file))

    #result = subprocess.run([
    #    'python3', index_py_path, index_path
    #], input=query, text=True, capture_output=True)

    # Use python3 for compatibility
    result = subprocess.run([
        sys.executable, search_py_path, index_path
    ], input=query, text=True, capture_output=True)
    print(result)
    actual_output = parse_output(result.stdout)
    return expected_output, actual_output

def main():
    test_dir = os.getcwd()+'/test'
    index_py_path = os.getcwd()+'/index.py'
    search_py_path = os.getcwd()+'/search.py'
    index_path = os.getcwd()+'/doc_index'
    results = []
    for i in range(1, 11):
        query_file = os.path.join(test_dir, f'query{i}.txt')
        answer_file = os.path.join(test_dir, f'answer{i}.txt')
        if not os.path.exists(query_file) or not os.path.exists(answer_file):
            if not os.path.exists(query_file):
                print(f"query file not exists {query_file}")
            if not os.path.exists(answer_file):
                print(f"answer file not exists {answer_file}")
            continue
        expected, actual = run_test(query_file, answer_file,index_py_path, search_py_path, index_path)
        if i == 1:
            print(f"Expected: {expected}")
            print(f"Got: {actual}")
        if i in [8, 9, 10]:
            correct = expected == actual
            print(f'Query {i}:', 'CORRECT' if correct else 'INCORRECT')
        else:
            precision, recall, fscore = compute_metrics(expected, actual)
            print(f'Query {i}: Precision={precision:.2f}, Recall={recall:.2f}, F-score={fscore:.2f}')
        results.append((i, expected, actual))

if __name__ == "__main__":
    main()
