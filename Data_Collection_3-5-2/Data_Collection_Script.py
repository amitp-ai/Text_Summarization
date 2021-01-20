import requests
import os
import subprocess
import hashlib

'''
Assumes Linux/Bash OS for tgz extraction (using subprocess.run())

The CNN dataset has 92579 examples and is 392.4MB in size.
The CNN dataset md5 checksum is: 85ac23a1926a831e8f46a6b8eaf57263

The Daily Mail dataset has 219506 examples and is 979.2MB in size.
The Daily Mail dataset md5 checksum is: f9c5f565e8abe86c38bfa4ae8f96fd72

The Gigaword dataset has 3993608 examples and is 939.4MB in size.
The Gigaword dataset md5 checksum is: 064e658dc0d42e0f721d03287a7a9913

'''

def download_file_from_google_drive(id_, destination):
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


    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id_ }, stream = True)
    token = get_confirm_token(response)

    if token:
      params = { 'id' : id_, 'confirm' : token }
      response = session.get(URL, params = params, stream = True)
  
    save_response_content(response, destination)    


def find_size_of_dir(path_dir):
    totsize= 0
    for dirpath, dirnames, filenames in os.walk(path_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            totsize += os.path.getsize(fp)
    return totsize

def get_checksum(file_name):
    # Open,close, read file and calculate MD5 on its contents 
    with open(file_name, "rb") as f:
        # read contents of the file
        data = f.read()    
        # pipe contents of the file through
        md5_returned = hashlib.md5(data).hexdigest()
    return md5_returned

if __name__ == '__main__':


    # exit() #to skip the below code
    
    #get CNN dataset
    id_ = '0BwmD_VLjROrfTHk4NFg2SndKcjQ'
    destination = './cnn.tgz'
    download_file_from_google_drive(id_, destination[2:])
    checksum = get_checksum(destination)
    subprocess.run(args=['tar', 'xzf', destination])
    subprocess.run(args=['rm', '-f', destination])
    print('Done Unzipping')
    data = len(os.listdir('./cnn/stories/'))
    totsize = find_size_of_dir('./cnn')
    print(f'The CNN dataset has {data} examples and is {totsize/1e6:.1f}MB in size.')
    print(f'The CNN dataset md5 checksum is: {checksum}\n')

    #get Daily mail dataset
    id_ = '0BwmD_VLjROrfM1BxdkxVaTY2bWs'
    destination = './dailymail.tgz'
    download_file_from_google_drive(id_, destination[2:])
    checksum = get_checksum(destination)
    subprocess.run(args=['tar', 'xzf', destination])
    subprocess.run(args=['rm', '-f', destination])
    print('Done Unzipping')
    data = len(os.listdir('./dailymail/stories'))
    totsize = find_size_of_dir('./dailymail')
    print(f'The Daily Mail dataset has {data} examples and is {totsize/1e6:.1f}MB in size.')
    print(f'The Daily Mail dataset md5 checksum is: {checksum}\n')

    #Gigaword dataset
    id_ = '0B6N7tANPyVeBNmlSX19Ld2xDU1E'
    destination = './gigaword.tar.gz'
    download_file_from_google_drive(id_, destination[2:])
    checksum = get_checksum(destination)
    subprocess.run(args=['tar', 'xzf', destination])
    subprocess.run(args=['rm', '-f', destination])
    print('Done Unzipping')
    os.rename('sumdata', 'gigaword')
    subprocess.run(args=['gzip', '-dr', './gigaword/train'])
    with open('./gigaword/train/train.title.txt') as f:
        gigword_num_examples = 0
        for l in f:
            gigword_num_examples += 1
    with open('./gigaword/train/valid.title.filter.txt') as f:
        for l in f:
            gigword_num_examples += 1
    totsize = find_size_of_dir('./gigaword')
    print(f'The Gigaword dataset has {gigword_num_examples} examples and is {totsize/1e6:.1f}MB in size.')
    print(f'The Gigaword dataset md5 checksum is: {checksum}\n')
