import requests
import os
import tqdm
def download_file(target_path,url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    file_downloaded = False
    file_path = os.path.join(target_path,local_filename)
    byte_pos = 0
    try:
        os.remove(file_path)
    except OSError:
        pass
    while not file_downloaded:
        resume_header = {f'Range': 'bytes=%d-' % byte_pos}
        try:
            with requests.get(url, headers=resume_header, stream=True,  verify=False, allow_redirects=True) as r:
            #with requests.get(url, stream=True) as r:
                r.raise_for_status()
                for chunk in  r.iter_content(chunk_size=8192):
                    with open(file_path, 'ab') as f:
                        # If you have chunk encoded response uncomment if
                        # and set chunk_size parameter to None.
                        #if chunk: 
                        f.write(chunk)
                        byte_pos += 1
                file_downloaded = True
        except:
            print('An error occured while downloading. Retrying...')
    return local_filename

urls = [
#'https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz'
'https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs-lookup.tsv.gz'
,'https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz'
,'https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz'
,'https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-top100.gz'
,'https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz'
,'https://msmarco.blob.core.windows.net/msmarcoranking/docleaderboard-queries.tsv.gz'
,'https://msmarco.blob.core.windows.net/msmarcoranking/docleaderboard-top100.tsv.gz'
]

for url in urls:
    print(f'Getting file from {url}')
    download_file('tst',url)