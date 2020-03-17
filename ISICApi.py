import requests
import os

class ISICApi(object):
    def __init__(self, hostname='https://isic-archive.com', username=None, password=None):
        self.baseUrl = f'{hostname}/api/v1'
        self.authToken = None
        if username is not None:
            if password is None:
                password = input(f'Password for user "{username}":')
            self.authToken = self._login(username, password)

    '''
        appends the base url to the start of the endpoint
    '''
    def _makeUrl(self, endpoint):
        return f'{self.baseUrl}/{endpoint}'

    '''
        login to the ISIC API
    '''
    def _login(self, username, password):
        authResponse = requests.get(self._makeUrl('user/authentication'), auth=(username, password))
        # error while loggin in
        if not authResponse.ok:
            raise Exception(f'Login error: {authResponse.json()["message"]}')
        # login success
        authToken = authResponse.json()['authToken']['token']
        return authToken

    def get(self, endpoint):
        url = self._makeUrl(endpoint)
        headers = {'Girder-Token': self.authToken} if self.authToken else None
        return requests.get(url, headers=headers)

    def getJson(self, endpoint):
        return self.get(endpoint).json()

    def getJsonList(self, endpoint):
        endpoint += '&' if '?' in endpoint else '?'
        LIMIT = 50
        offset = 0
        while True:
            resp = self.get(
                f'{endpoint}limit={LIMIT:d}&offset={offset:d}'
            ).json()
            if not resp:
                break
            for elem in resp:
                yield elem
            offset += LIMIT

    '''
        POST the API
    '''
    def post(self, endpoint, params=None):
        url = self._makeUrl(endpoint)
        headers = {'Girder-Token': self.authToken} if self.authToken else None
        return requests.post(url, headers=headers, data=params)
    
    '''
        downloads an image from the API
    '''
    def downloadSegmentationMask(self, imageId, imgName, dst):
        if not os.path.exists(dst):
            os.makedirs(dst)
        segmentationList = self.getJson('segmentation?imageId='+imageId)
        print('Downloading %s images' % len(segmentationList))
        for segmentation in segmentationList:
            print(segmentation['_id'])
            imageFileResp = self.get('segmentation/%s/mask' % segmentation['_id'])
            imageFileResp.raise_for_status()
            imageFileOutputPath = os.path.join(dst, imgName+'.bmp')
            with open(imageFileOutputPath, 'wb') as imageFileOutputStream:
                for chunk in imageFileResp:
                    imageFileOutputStream.write(chunk)
            break;