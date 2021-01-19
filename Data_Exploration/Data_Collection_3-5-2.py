import shutil
import requests
import os

'''
The CNN dataset has 92579 examples and is 392.367704MB in size.
The Daily Mail dataset has 219506 examples and is 979.205688MB in size.
The Gigaword dataset has 3993608 examples and is 939.402629MB in size.
'''

#This code is copied from stack overflow <https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive>
def download_file_from_google_drive(id_, destination):
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


def unpack_file(file_name):
    extract_path = './'
    shutil.unpack_archive(file_name, extract_path)

def find_size_of_dir(path_dir):
    totsize= 0
    for dirpath, dirnames, filenames in os.walk(path_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            totsize += os.path.getsize(fp)
    return totsize

if __name__ == '__main__':
    exit()
    
    #get CNN dataset
    id_ = '0BwmD_VLjROrfTHk4NFg2SndKcjQ'
    destination = 'cnn.tgz'
    download_file_from_google_drive(id_, destination)
    unpack_file(destination)
    os.remove(destination)
    data = len(os.listdir('./cnn/stories/'))
    totsize = find_size_of_dir('./cnn')
    print(f'The CNN dataset has {data} examples and is {totsize/1e6}MB in size.')

    #get Daily mail dataset
    id_ = '0BwmD_VLjROrfM1BxdkxVaTY2bWs'
    destination = 'dailymail.tgz'
    download_file_from_google_drive(id_, destination)
    unpack_file(destination)
    os.remove(destination)
    data = len(os.listdir('./dailymail/stories'))
    totsize = find_size_of_dir('./dailymail')
    print(f'The Daily Mail dataset has {data} examples and is {totsize/1e6}MB in size.')

    #Gigaword dataset
    id_ = '0B6N7tANPyVeBNmlSX19Ld2xDU1E'
    destination = 'gigaword.tar.gz'
    download_file_from_google_drive(id_, destination)
    unpack_file(destination)
    os.remove(destination)
    with open('./gigaword/train/train.title.txt') as f:
        gigword_num_examples = 0
        for l in f:
            gigword_num_examples += 1
    with open('./gigaword/train/valid.title.filter.txt') as f:
        for l in f:
            gigword_num_examples += 1
    totsize = find_size_of_dir('./gigaword')
    print(f'The Gigaword dataset has {gigword_num_examples} examples and is {totsize/1e6}MB in size.')
