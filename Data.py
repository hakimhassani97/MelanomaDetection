import os
import shutil
import urllib
import json
import requests
from ISICApi import ISICApi

class Data:
    @staticmethod
    def loadFilesAsArray(folder):
        files=os.listdir(folder)
        return files

    '''
        copies the PH2 dataset to project/data folder
    '''
    @staticmethod
    def loadImagesToFolder():
        src='D:/HAKIM/MIV M2/PFE/data/PH2Dataset/PH2 Dataset images/'
        dst='D:/HAKIM/MIV M2/PFE/project/data/PH2/images/'
        folders=os.listdir(src)
        for folder in folders:
            s=src+folder+'/'+folder+'_Dermoscopic_Image'
            img=os.listdir(s)[0]
            if len(os.listdir(src+folder+'/'+folder+'_roi'))>0 :
                dstImg='1_'+img
            else:
                dstImg='0_'+img
            shutil.copy2(s+'/'+img,dst+dstImg)
    
    '''
        downloads the ISIC dataset from ISIC's API
    '''
    @staticmethod
    def downloadISIC():
        # request.urlretrieve("http://www.gunnerkrigg.com//comics/00000001.jpg", "00000001.jpg")
        API_URL='https://isic-archive.com/api/v1/image'
        dst='D:/HAKIM/MIV M2/PFE/project/data/ISIC/images/'
        # login to ISIC API
        username = 'hakimhassani97@gmail.com'
        password = '@Kimo1234'
        api = ISICApi(username=username, password=password)
        # set API params
        limit=50
        params={'limit':50}
        params = urllib.parse.urlencode(params)
        # get request the ISIC API
        results=urllib.request.urlopen(API_URL+'?%s'%params)
        results=results.read()
        # convert result to JSON
        results=json.loads(results)
        # loop throught JSON
        for i,result in enumerate(results):
            print(str(i),result)
            # urllib.request.urlretrieve(API_URL+'/'+result['_id']+'/download', dst+result['name']+".bmp")


Data.downloadISIC()