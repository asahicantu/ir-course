processed_indexes = []
continue_indexing = False
doc_id_cache = 'D643193'
with open('MS-MARCO/required_docs.tsv','r') as dump:
    for line in dump:
        
        doc = extract_document(line)
        if doc[0] == doc_id_cache:
            continue_indexing = True
        
        if continue_indexingindexing:
            REMOTE_ES.index(index=INDEX_NAME, id=doc[0], body=doc[1])
            print(f"\rIndexing: {len(processed_indexes)} {doc[0]}...", end="")
        processed_indexes.append(doc[0])
        




processed_indexes = []
continue_indexing = False
doc_id_cache = 'D3095996'
#Indexing: 27262 D3095996...
with open('MS-MARCO/required_docs.tsv','r') as dump:
    for line in dump:
        doc = extract_document(line)
        if doc[0] == doc_id_cache:
            continue_indexing = True
        
        if continue_indexing:
            REMOTE_ES.index(index=INDEX_NAME, id=doc[0], body=doc[1])
            print(f"\rIndexing: {len(processed_indexes)} {doc[0]}...", end="")
        processed_indexes.append(doc[0])
        



rocessed_indexes = []
continue_indexing = False
doc_id_cache = 'D3095996'
#Indexing: 27262 D3095996...
with gzip.GzipFile('MS-MARCO/required_docs.tsv.gz','rb') as dump:
    for line in dump:
        line = line.decode('UTF-8')
        doc = extract_document(line)
        if doc[0] == doc_id_cache:
            continue_indexing = True
        
        if continue_indexing:
            REMOTE_ES.index(index=INDEX_NAME, id=doc[0], body=doc[1])
            print(f"\rIndexing: {len(processed_indexes)} {doc[0]}...", end="")
        processed_indexes.append(doc[0])
        