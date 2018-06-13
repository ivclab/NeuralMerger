"""
Modification of
- https://github.com/carpedm20/BEGAN-tensorflow/blob/master/download.py
- http://stackoverflow.com/a/39225039
"""
import requests
import zipfile
import os
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

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

if __name__ == "__main__":
    
    file_id = '1clKF1zxd2lToejzKvzuubWJIkt-xpoIk'
    destination = './weight_loader/weight.zip'
    download_file_from_google_drive(file_id, destination)
    with zipfile.ZipFile(destination) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall('./weight_loader')
    os.remove(destination)

    file_id = '1y0gfVkg44BlN8gbJgr8azJq115CaiLJy'
    destination = 'tfdata.zip'
    download_file_from_google_drive(file_id, destination)
    with zipfile.ZipFile(destination) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall('./')
    os.remove(destination)