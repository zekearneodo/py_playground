import io
import sys
import logging
import simplejson as json

from apiclient.http import MediaIoBaseDownload
from googleapiclient import errors
import hashlib

# TODO: -retry wrapper doesn' work


logger = logging.getLogger("googleapi.filehandling")


def md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def retry_wrapper(default_retries=5, default_catch_codes=[403, 503]):
    '''
    A doble wrapper to make a function retry defautl_retries if a code error is found.
    (This is to prevent things from breaking if google resources are temporally unavailable.
    :param default_retries:
    :param default_catch_codes:
    :return:
    '''
    def wrap(service_function):
        def persistent_function(*args, **kwargs):
            if 'retries' in kwargs.keys():
                retries = kwargs['retries']
            else:
                retries = default_retries
            if 'codes' in kwargs.keys():
                catch_codes = default_catch_codes

            logger.debug('retries {}'.format(retries))
            for n in range(retries):
                try:
                    return_value = service_function(*args, **kwargs)
                    return return_value
                except errors.HttpError as err:
                    error = json.loads(err.content)
                    if error.get('code') in catch_codes:
                        logger.debug('Error {0}, waiting to retry {1}'.format(error.get('code'), n))
                        time.sleep(2**n)
                    else:
                        raise
            logger.warning('Critical error')
            return None
        return persistent_function
    return wrap


@retry_wrapper
def verify(google_file, local_file_path):
    return google_file['md5Checksum'] == md5(local_file_path)


def find_children(parent_id, query, *args, **kwargs):
    full_query = "\'{}\' in parents".format(parent_id)
    if query is not None:
        full_query+= " and " + query
    print(full_query)
    return service.files().list(q=full_query, *args, **kwargs).execute()

def get_file_item(service, file_id, retries=5):
    for n in range(retries):
        try:
            file_item = service.files().get(id=file_id).execute()
            return file_item
        except errors.HttpError, e:
            error = json.loads(e.content)
            if error.get('code') == 403 or error.get('code') == 503:
                logger.warning('Error  {}'.format(error.get('errors')[0].get('reason')))
                time.sleep((2**n))
            else:
                raise
    logger.warning('Error in get_file_item')
    return None


def download_chunk(downloader, retries=5):
    for n in range(retries):
        try:
            status, done = downloader.next_chunk()
            return status, done
        except errors.HttpError, e:
            error = json.loads(e.content)
            if error.get('code') == 403 or error.get('code') == 503:
                logger.warning('Error  {}'.format(error.get('errors')[0].get('reason')))
                time.sleep((2**n))
            else:
                raise
        logger.warning('Error in download_file_chunk')


def download_file(service, file_id, retries=5):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_name, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    last_progress = 0
    progress_step = 0.05

    while done is False:
        try:
            status, done = downloader.next_chunk()
            if status.progress() > last_progress:
                logger.debug('Downloaded {}%'.format(int(status.progress() * 100)))
                last_progress += progress_step
                sys.stdout.flush()
        except errors.HttpError as err:
            print(err)
            if int(err.resp['status']) == 503:
                logger.warning('Sevice Unavailable, retrying')
            else:
                raise

    logger.info('Download complete')
    return True




class google_dvive_service():
    def __init__(self, credentials, max_retries=5):
        http = credentials.authorize(httplib2.Http())
        self.service = discovery.build('drive', 'v2', http=http)
        self.max_retries = max_retries
        self.retries = 0

    def download_file(self, file_item, dest_path):
        pass


