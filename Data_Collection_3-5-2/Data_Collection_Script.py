import requests
import os
import subprocess
import hashlib

'''
Assumes Linux/Bash OS for tgz extraction (using subprocess.run())

The CNN dataset has 92579 examples and is 392.4MB in size.
The CNN dataset md5 checksum (for *.tgz) is: 85ac23a1926a831e8f46a6b8eaf57263

The Daily Mail dataset has 219506 examples and is 979.2MB in size.
The Daily Mail dataset md5 checksum (for *.tgz) is: f9c5f565e8abe86c38bfa4ae8f96fd72

The Gigaword dataset has 3993608 examples and is 939.4MB in size.
The Gigaword dataset md5 checksum (for *.tar.gz) is: 064e658dc0d42e0f721d03287a7a9913

The Arxiv dataset has 4 examples and is 15150.2MB in size.
The Arxiv dataset md5 checksum (for *.zip) is: 6242aaf5cfcc7814473eee8b779c1b9f

The Pubmed dataset has 4 examples and is 4940.0MB in size.
The Pubmed dataset md5 checksum (for *.zip) is: 3ae396f2b690253e7379a038b410300c
'''   

def download_file(destination, id_=None, url=None):
#This funciton is copied from stack overflow <https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive>
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    params = {}
    if id_ and not url: #i.e. from Google Drive
        url = "https://docs.google.com/uc?export=download"
        params = { 'id' : id_ }

    session = requests.Session()
    response = session.get(url, params = params, stream = True)
    token = get_confirm_token(response)

    if token:
        params = {'confirm' : token }
        if id_:
            params = { 'id' : id_, 'confirm' : token }
    response = session.get(url, params = params, stream = True)  
    save_response_content(response, destination)    


def find_size_of_dir(path_dir):
    totsize= 0
    for dirpath, dirnames, filenames in os.walk(path_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            totsize += os.path.getsize(fp)
    return totsize

def find_num_examples_in_dir(path_dir):
    totsize= 0
    for dirpath, dirnames, filenames in os.walk(path_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            totsize += find_num_examples(fp)
    return totsize

def find_num_examples(file_name):
    with open(file_name) as f:
        num_examples = 0
        for l in f:
            num_examples += 1
    return num_examples

# def delete_get_checksum(file_name):
#     #this doesn't work for large files. runs into memory error.
#     # Open,close, read file and calculate MD5 on its contents 
#     with open(file_name, "rb") as f:
#         # read contents of the file
#         data = f.read()    
#         # pipe contents of the file through
#         md5_returned = hashlib.md5(data).hexdigest()
#     return md5_returned

def get_checksum(file_name, block_size=128*64):
    # Open in blocks to fit in memory,close, read file and calculate MD5 on its contents 
    md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        while True:
            # read contents of the file
            chunk = f.read(block_size)
            if not chunk: break
            md5.update(chunk)
    # pipe contents of the file through
    md5_returned = md5.hexdigest()
    return md5_returned

if __name__ == '__main__':
    
    # exit() #to skip the below code
    
    #get BigPatent
    print('Geting BigPatent Data...')
    id_ = '1J3mucMFTWrgAYa3LuBZoLRR3CzzYD3fa'
    destination = './bigPatentData.tar.gz'
    download_file(destination, id_=id_)
    checksum = get_checksum(destination)
    subprocess.run(args=['tar', 'xzf', destination])
    subprocess.run(args=['rm', '-f', destination])
    print('Done Unzipping')
    subprocess.run(args=['tar', 'xzf', './bigPatentData/train.tar.gz', '-C', './bigPatentData'])
    subprocess.run(args=['rm', '-f', './bigPatentData/train.tar.gz'])
    subprocess.run(args=['gzip', '-dr', './bigPatentData/train']) #recursively unzip all files in this folder and delete the .gz files
    subprocess.run(args=['tar', 'xzf', './bigPatentData/val.tar.gz', '-C', './bigPatentData'])
    subprocess.run(args=['rm', '-f', './bigPatentData/val.tar.gz'])
    subprocess.run(args=['gzip', '-dr', './bigPatentData/val']) #recursively unzip all files in this folder and delete the .gz files
    subprocess.run(args=['tar', 'xzf', './bigPatentData/test.tar.gz', '-C', './bigPatentData'])
    subprocess.run(args=['rm', '-f', './bigPatentData/test.tar.gz'])
    subprocess.run(args=['gzip', '-dr', './bigPatentData/test']) #recursively unzip all files in this folder and delete the .gz files
    print('Done Unzipping all the files inside the main data folder')
    data = find_num_examples_in_dir('./bigPatentData/train')
    totsize = find_size_of_dir('./bigPatentData')
    print(f'The BigPatent dataset has {data} training examples and is {totsize/1e6:.1f}MB in size.')
    print(f'The BigPatent dataset md5 checksum (for *.tar.gz) is: {checksum}\n')
    
    
    #get Scientific Papers (ArXiv & PubMed)
    print('Geting Arxiv Data...')
    url = 'https://archive.org/download/armancohan-long-summarization-paper-code/arxiv-dataset.zip'
    destination = './arxiv-dataset.zip'
    download_file(destination, url=url)
    checksum = get_checksum(destination)
    print('Running Unzip...')
    subprocess.run(args=['unzip', '-q', destination])
    print('Done Unzipping')
    data = find_num_examples('./arxiv-dataset/train.txt')
    totsize = find_size_of_dir('./arxiv-dataset')
    print(f'The Arxiv dataset has {data} training examples and is {totsize/1e6:.1f}MB in size.')
    print(f'The Arxiv dataset md5 checksum (for *.zip) is: {checksum}\n')
    subprocess.run(args=['rm', '-f', destination])
    subprocess.run(args=['rm', '-rf', './__MACOSX'])

    print('Geting Pubmed Data...')
    url = 'https://archive.org/download/armancohan-long-summarization-paper-code/pubmed-dataset.zip'
    destination = './pubmed-dataset.zip'
    download_file(destination, url=url)
    checksum = get_checksum(destination)
    print('Running Unzip...')
    subprocess.run(args=['unzip', '-q', destination])
    print('Done Unzipping')
    data = find_num_examples('./pubmed-dataset/train.txt')
    totsize = find_size_of_dir('./pubmed-dataset')
    print(f'The Pubmed dataset has {data} training examples and is {totsize/1e6:.1f}MB in size.')
    print(f'The Pubmed dataset md5 checksum (for *.zip) is: {checksum}\n')
    subprocess.run(args=['rm', '-f', destination])
    
    #get CNN dataset
    print('Geting CNN Data...')
    id_ = '0BwmD_VLjROrfTHk4NFg2SndKcjQ'
    destination = './cnn.tgz'
    download_file(destination, id_=id_)
    checksum = get_checksum(destination)
    subprocess.run(args=['tar', 'xzf', destination])
    subprocess.run(args=['rm', '-f', destination])
    print('Done Unzipping')
    data = len(os.listdir('./cnn/stories/'))
    totsize = find_size_of_dir('./cnn')
    print(f'The CNN dataset has {data} examples and is {totsize/1e6:.1f}MB in size.')
    print(f'The CNN dataset md5 checksum (for *.tgz) is: {checksum}\n')

    #get Daily mail dataset
    print('Geting Daily Mail Data...')
    id_ = '0BwmD_VLjROrfM1BxdkxVaTY2bWs'
    destination = './dailymail.tgz'
    download_file(destination, id_=id_)
    checksum = get_checksum(destination)
    subprocess.run(args=['tar', 'xzf', destination])
    subprocess.run(args=['rm', '-f', destination])
    print('Done Unzipping')
    data = len(os.listdir('./dailymail/stories'))
    totsize = find_size_of_dir('./dailymail')
    print(f'The Daily Mail dataset has {data} examples and is {totsize/1e6:.1f}MB in size.')
    print(f'The Daily Mail dataset md5 checksum (for *.tgz) is: {checksum}\n')

    #Gigaword dataset
    print('Geting Gigaword Data...')
    id_ = '0B6N7tANPyVeBNmlSX19Ld2xDU1E'
    destination = './gigaword.tar.gz'
    download_file(destination, id_=id_)
    checksum = get_checksum(destination)
    subprocess.run(args=['tar', 'xzf', destination])
    subprocess.run(args=['rm', '-f', destination])
    print('Done Unzipping')
    os.rename('sumdata', 'gigaword')
    subprocess.run(args=['gzip', '-dr', './gigaword/train']) #recursively unzip all files in this folder and delete the .gz files
    gigword_num_examples = find_num_examples('./gigaword/train/train.title.txt')
    gigword_num_examples += find_num_examples('./gigaword/train/valid.title.filter.txt')
    totsize = find_size_of_dir('./gigaword')
    print(f'The Gigaword dataset has {gigword_num_examples} examples and is {totsize/1e6:.1f}MB in size.')
    print(f'The Gigaword dataset md5 checksum (for *.tar.gz) is: {checksum}\n')

    