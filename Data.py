import os
import shutil

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
Data.loadImagesToFolder()