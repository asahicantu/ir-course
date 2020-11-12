import elasticsearch
import math
import numpy as np
import os
import pytest
import random
import requests
import tarfile

from collections import Counter
from collections import defaultdict
from elasticsearch import Elasticsearch


# YOUR CODE HERE
from sklearn.ensemble import RandomForestRegressor


def analyze_query(es, query, field, index='toy_index'):
    """Analyzes a query with respect to the relevant index. 
    
    Arguments:
        es: Elasticsearch object instance.
        query: String of query terms.
        field: The field with respect to which the query is analyzed. 
        index: Name of the index with respect to which the query is analyzed.  
    
    Returns:
        A list of query terms that exist in the specified field among the documents in the index. 
    """
    tokens = es.indices.analyze(index=index, body={'text': query})['tokens']
    query_terms = []
    for t in sorted(tokens, key=lambda x: x['position']):
        ## Use a boolean query to find at least one document that contains the term.
        hits = es.search(index=index, body={'query': {'match': {field: t['token']}}}, 
                                   _source=False, size=1).get('hits', {}).get('hits', {})
        doc_id = hits[0]['_id'] if len(hits) > 0 else None
        if doc_id is None:
            continue
        query_terms.append(t['token'])
    return query_terms

    def get_doc_term_freqs(es, doc_id, field, index='toy_index'):
    """Gets the term frequencies of a field of an indexed document. 
    
    Arguments:
        es: Elasticsearch object instance.
        doc_id: Document identifier with which the document is indexed. 
        field: Field of document to consider for term frequencies.
        index: Name of the index where document is indexed.  
    
    Returns:
        Dictionary of terms and their respective term frequencies in the field and document.  
    """
    tv = es.termvectors(index=index, id=doc_id, fields=field, term_statistics=True)
    if tv['_id'] != doc_id:
        return None
    if field not in tv['term_vectors']:
        return None
    term_freqs = {}
    for term, term_stat in tv['term_vectors'][field]['terms'].items():
        term_freqs[term] = term_stat['term_freq']
    return term_freqs
        
def get_query_term_freqs(es, query_terms):
    """Gets the term frequencies of a list of query terms. 
    
    Arguments:
        es: Elasticsearch object instance.
        query_terms: List of query terms, analyzed using `analyze_query` with respect to some relevant index. 
    
    Returns:
        A list of query terms that exist in the specified field among the documents in the index. 
    """
    c = Counter()
    for term in query_terms:
        c[term] += 1
    return dict(c)

def extract_query_features(query_terms, es, index='toy_index'):
    """Extracts features of a query.

        Arguments:
            query_terms: List of analyzed query terms.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch service.
        Returns:
            Dictionary with keys 'query_length', 'query_sum_idf', 'query_max_idf', and 'query_avg_idf'.
    """
    q_features = {}

    # Implement `query_length` feature.
    # YOUR CODE HERE

    if len(query_terms) == 0:
        q_features['query_length'] = 0
        q_features['query_sum_idf'] = 0
        q_features['query_max_idf'] = 0
        q_features['query_avg_idf'] = 0
        return q_features

    q_features['query_length'] = len(query_terms)

    # Implement `query_{sum|max|avg}_idf` features.
    # YOUR CODE HERE

    count_docs_with_term = []
    total_docs_in_index = int(es.cat.count(index=index, params={"format": "json"})[0]['count'])

    for query in query_terms:
        res = es.search(index=index, _source=False, body={
            'query':
                {'match':
                     {'body': query}
                 }
        })['hits']['total']['value']
        count_docs_with_term.append(res)

    q_features['query_sum_idf'] = sum([np.log(total_docs_in_index / freq) for freq in count_docs_with_term])
    q_features['query_max_idf'] = max([np.log(total_docs_in_index / freq) for freq in count_docs_with_term])
    q_features['query_avg_idf'] = np.mean([np.log(total_docs_in_index / freq) for freq in count_docs_with_term])

    return q_features


def extract_doc_features(doc_id, es, index='toy_index'):
    """Extracts features of a document.
    
        Arguments:
            doc_id: Document identifier of indexed document.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch service. 

        Returns:
            Dictionary with keys 'doc_length_title', 'doc_length_body'.
    """
    doc_features = {}

    # YOUR CODE HERE

    terms = get_doc_term_freqs(es, doc_id, 'body', index)
    if terms is None:
        doc_features['doc_length_body'] = 0
    else:
        doc_features['doc_length_body'] = sum(terms.values())

    terms = get_doc_term_freqs(es, doc_id, 'title', index)
    if terms is None:
        doc_features['doc_length_title'] = 0
    else:
        doc_features['doc_length_title'] = sum(terms.values())

    return doc_features

def extract_query_doc_features(query_terms, doc_id, es, index='toy_index'):
    """Extracts features of a query and document pair.
    
        Arguments:
            query_terms: List of analyzed query terms.
            doc_id: Document identifier of indexed document.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch service. 
            
        Returns:
            Dictionary with keys 'unique_query_terms_in_title', 'unique_query_terms_in_body',
            'sum_TF_title', 'sum_TF_body', 'max_TF_title', 'max_TF_body', 'avg_TF_title', 'avg_TF_body'. 
    """
    q_doc_features = {}

    # YOUR CODE HERE

    if len(query_terms) == 0:
        q_doc_features['unique_query_terms_in_title'] = 0
        q_doc_features['unique_query_terms_in_body'] = 0
        q_doc_features['sum_TF_body'] = 0
        q_doc_features['max_TF_body'] = 0
        q_doc_features['avg_TF_body'] = 0
        q_doc_features['sum_TF_title'] = 0
        q_doc_features['max_TF_title'] = 0
        q_doc_features['avg_TF_title'] = 0
        return q_doc_features

    terms_title = get_doc_term_freqs(es, doc_id, 'title', index)
    terms_body = get_doc_term_freqs(es, doc_id, 'body', index)

    def agg(terms_dict, query_terms_list, func):
        freq_list = []
        for term in query_terms_list:
            if term in terms_dict.keys():
                freq_list.append(terms_dict[term])
            else:
                freq_list.append(0)
        return func(freq_list)

    if terms_title is None:
        q_doc_features['sum_TF_title'] = 0
        q_doc_features['max_TF_title'] = 0
        q_doc_features['avg_TF_title'] = 0
    else:
        q_doc_features['sum_TF_title'] = agg(terms_title, query_terms, sum)
        q_doc_features['max_TF_title'] = agg(terms_title, query_terms, max)
        q_doc_features['avg_TF_title'] = agg(terms_title, query_terms, np.mean)

    if terms_body is None:
        q_doc_features['sum_TF_body'] = 0
        q_doc_features['max_TF_body'] = 0
        q_doc_features['avg_TF_body'] = 0
    else:
        q_doc_features['sum_TF_body'] = agg(terms_body, query_terms, sum)
        q_doc_features['max_TF_body'] = agg(terms_body, query_terms, max)
        q_doc_features['avg_TF_body'] = agg(terms_body, query_terms, np.mean)

    # UNIQUE QUERY TERMS
    query_terms = set(query_terms)
    if terms_title is None:
        q_doc_features['unique_query_terms_in_title'] = 0
    else:
        q_doc_features['unique_query_terms_in_title'] = len([t for t in query_terms if t in terms_title.keys()])
    if terms_body is None:
        q_doc_features['unique_query_terms_in_body'] = 0
    else:
        q_doc_features['unique_query_terms_in_body'] = len([t for t in query_terms if t in terms_body.keys()])

    return q_doc_features

FEATURES_QUERY = ['query_length', 'query_sum_idf', 'query_max_idf', 'query_avg_idf']
FEATURES_DOC = ['doc_length_title', 'doc_length_body']
FEATURES_QUERY_DOC = ['unique_query_terms_in_title', 'sum_TF_title', 'max_TF_title', 'avg_TF_title',
                      'unique_query_terms_in_body', 'sum_TF_body', 'max_TF_body', 'avg_TF_body'
                     ]

def extract_features(query_terms, doc_id, es, index='toy_index'):
    """Extracts query features, document features and query-document features of a query and document pair.
    
        Arguments:
            query_terms: List of analyzed query terms.
            doc_id: Document identifier of indexed document.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch service. 
            
        Returns:
            List of extracted feature values in a fixed order.
    """
    feature_vect = []
    
    query_features = extract_query_features(query_terms, es, index=index)
    for f in FEATURES_QUERY:
        feature_vect.append(query_features[f])
    
    doc_features = extract_doc_features(doc_id, es, index=index)
    for f in FEATURES_DOC:
        feature_vect.append(doc_features[f])

    query_doc_features = extract_query_doc_features(query_terms, doc_id, es, index=index)
    for f in FEATURES_QUERY_DOC:
        feature_vect.append(query_doc_features[f])

    return feature_vect

def load_queries(filepath):
    """Loads queries from a file.
    
        Arguments:
            filepath: String (constructed using os.path) of the filepath to a file with queries.  
            
        Returns:
            A dictionary with query IDs and corresponding query strings. 
    """
    queries = {}

    # YOUR CODE HERE

    with open(filepath, 'r') as file:
        query_id = ''
        query_text = ''
        for line in file.readlines():
            words_list = line.strip().split(' ')
            if len(words_list) < 2:
                continue
            if words_list[0] == '<num>':
                query_id = words_list[2]
            if words_list[0] == '<title>':
                query_text = ' '.join(words_list[1:])
                queries[query_id] = query_text

    return queries


def load_qrels(filepath):
    """Loads query relevance judgments from a file.
    
        Arguments:
            filepath: String (constructed using os.path) of the filepath to a file with queries.  
            
        Returns:
            A dictionary with query IDs and a corresponding list of document IDs for documents judged 
            relevant to the query. 
    """
    qrels = defaultdict(list)

    # YOUR CODE HERE

    with open(filepath, 'r') as file:
        for line in file.readlines():
            words_list = line.split()
            if len(words_list) < 2:
                continue
            qrels[words_list[0]].append(words_list[1])

    return qrels

random.seed(a=1234567)
trec9_query_ids = sorted(list(TREC9_QUERIES.keys()))
random.shuffle(trec9_query_ids)
TRAIN_SIZE = int(len(trec9_query_ids) * 0.8)
TEST_SIZE = len(trec9_query_ids) - TRAIN_SIZE
TRAIN_QUERY_IDS = trec9_query_ids[:TRAIN_SIZE]
TEST_QUERY_IDS = trec9_query_ids[TRAIN_SIZE:]


def prepare_ltr_training_data(query_ids, es, index='trec9_index'):
    """Prepares feature vectors and labels for query and document pairs found in the training data.
    
        Arguments:
            query_ids: List of query IDs.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch service. 
            
        Returns:
            X: List of feature vectors extracted for each pair of query and retrieved or relevant document. 
            y: List of corresponding labels.
    """
    X = []
    y = []

    # YOUR CODE HERE


    for query_id in query_ids:
        relevent_docs = TREC9_QRELS[query_id]
        query = TREC9_QUERIES[query_id]
        analyzed_terms = analyze_query(es, query, 'body', index=index)
        for relevent_doc in relevent_docs:
            extracted_feature = extract_features(analyzed_terms, relevent_doc, es, index=index)
            X.append(extracted_feature)
            y.append(1)

        # try:
        hits = es.search(index='trec9_index', q=' '.join(analyzed_terms), _source=True, size=100)['hits']['hits']
        # except:
        #     continue

        for hit in hits:
            doc_id = hit['_id']
            if doc_id not in relevent_docs:
                extracted_feature = extract_features(analyzed_terms, doc_id, es, index=index)
                X.append(extracted_feature)
                y.append(0)

    return X[0:-1], y[0:-1]


    X_train, y_train = prepare_ltr_training_data(TRAIN_QUERY_IDS[:800], es, index='trec9_index')



    class PointWiseLTRModel(object):
    def __init__(self, regressor):
        """
        Arguments:
            classifier: An instance of scikit-learn regressor.
        """
        self.regressor = regressor

    def _train(self, X, y):
        """Trains an LTR model.
        
        Arguments:
            X: Features of training instances.
            y: Relevance assessments of training instances.
        """
        assert self.regressor is not None
        self.model = self.regressor.fit(X, y)

    def rank(self, ft, doc_ids):
        """Predicts relevance labels and rank documents for a given query.
        
        Arguments:
            ft: A list of feature vectors for query-document pairs.
            doc_ids: A list of document ids.
        Returns:
            List of tuples, each consisting of document ID and predicted relevance label.
        """
        assert self.model is not None
        rel_labels = self.model.predict(ft)
        sort_indices = np.argsort(rel_labels)[::-1]

        results = []
        for i in sort_indices:
            results.append((doc_ids[i], rel_labels[i]))
        return results

clf = RandomForestRegressor(max_depth=5, random_state=0)

# Instantiate PointWiseLTRModel.
ltr = PointWiseLTRModel(clf)
ltr._train(X_train, y_train)


# Find the predicted rankings of each test query.

def get_rankings(ltr, query_ids, es, index='trec9_index', rerank=False):
    """Generate rankings for each of the test query IDs.
    
        Arguments:
            ltr: A trained PointWiseLTRModel instance.
            query_ids: List of query IDs.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch service. 
            rerank: Boolean flag indicating whether the first-pass retrieval results should be reranked using the LTR model.
            
        Returns:
            A dictionary of rankings for each test query ID. 
    """

    test_rankings = {}
    for i, query_id in enumerate(query_ids):
        print('Processing query {}/{} ID {}'.format(i + 1, len(query_ids), query_id))
        # First-pass retrieval
        query_terms = analyze_query(es, TREC9_QUERIES[query_id], 'body', index=index)
        if len(query_terms) == 0:
            print('WARNING: query {} is empty after analysis; ignoring'.format(query_id))
            continue
        hits = es.search(index=index, q=' '.join(query_terms), _source=True, size=100)['hits']['hits']        
        test_rankings[query_id] = [hit['_id'] for hit in hits]
        
        # Rerank the first-pass result set using the LTR model.
        if rerank:
            # YOUR CODE HERE
            docs_to_rerank = test_rankings[query_id]
            ftr = [extract_features(query_terms, d, es, index=index) for d in docs_to_rerank]
            res = ltr.rank(ft=ftr, doc_ids=docs_to_rerank)
            ranked_docs = [d for d, score in res]
            test_rankings[query_id] = ranked_docs
    return test_rankings

    test_query_ids = TEST_QUERY_IDS[-100:]

    # Test of learning-to-rank rankings
rankings_ltr = get_rankings(ltr, test_query_ids, es, index='trec9_index', rerank=True)
assert len(rankings_ltr['MSH2691']) == 100
assert rankings_ltr['MSH2691'].index('87254618') < rankings_ltr['MSH2691'].index('87216812')


def get_reciprocal_rank(system_ranking, ground_truth):
    """Computes Reciprocal Rank (RR).
    
    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.
    
    Returns:
        RR (float).
    """
    for i, doc_id in enumerate(system_ranking):
        if doc_id in ground_truth:
            return 1 / (i + 1)
    return 0
    
def get_mean_eval_measure(system_rankings, ground_truth, eval_function):
    """Computes a mean of any evaluation measure over a set of queries.
    
    Args:
        system_rankings: Dict with query ID as key and a ranked list of document IDs as value.
        ground_truths: Dict with query ID as key and a set of relevant document IDs as value.
        eval_function: Callback function for the evaluation measure that mean is computed over.
    
    Returns:
        Mean evaluation measure (float).
    """
    sum_score = 0
    for query_id, system_ranking in system_rankings.items():
        sum_score += eval_function(system_ranking, ground_truth[query_id])
    return sum_score / len(system_rankings)