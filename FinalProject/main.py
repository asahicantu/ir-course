from elasticsearch import Elasticsearch
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestRegressor

USER = 'elastic'
PASS = 'IfKREtTr7fCqMYTD8NKE4yBi'
REMOTE_SERVER = f'https://{USER}:{PASS}@6a0fe46eef334fada72abc91933b54e8.us-central1.gcp.cloud.es.io:9243'
INDEX_NAME = 'ms-marco'

es = Elasticsearch(hosts=REMOTE_SERVER)


def analyze_query(es, query, field, index='ms-marco'):
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


def load_queries(filepath):
    """Loads queries from a file.

        Arguments:
            filepath: String (constructed using os.path) of the filepath to a file with queries.

        Returns:
            A dictionary with query IDs and corresponding query strings.
    """
    queries = {}
    with open(filepath, 'r', encoding="utf8") as file:
        for line in file:
            query_id, query_text = line.strip().split('\t')
            if query_text is not None:
                queries[int(query_id)] = query_text.strip()
    return queries


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

    if len(query_terms) == 0:
        q_features['query_length'] = 0
        q_features['query_sum_idf'] = 0
        q_features['query_max_idf'] = 0
        q_features['query_avg_idf'] = 0
        return q_features

    q_features['query_length'] = len(query_terms)

    count_docs_with_term = []
    total_docs_in_index = int(es.cat.count(index=index, params={"format": "json"})[0]['count'])

    for query in query_terms:
        res = es.count(index=index, body={
            'query':
                {'match':
                     {'body': query}
                 }
        })['count']
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


queries = load_queries('data/queries.doctrain.tsv')

query_terms = analyze_query(es, queries[564233], 'body', INDEX_NAME)
query = ' '.join(query_terms)

res = es.search(index=INDEX_NAME, _source=True, size=1, body={
            'query':
                {'match':
                     {'body': query}
                 }
        })['hits']

doc_id = res['hits'][0]['_id']
feature_vector = extract_features(query_terms, doc_id, es, index=INDEX_NAME)

print(feature_vector)