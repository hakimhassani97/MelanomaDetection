import cv2
import numpy as np
from Data import Data
from Contours import Contours

class Statistics:
    '''
        segments and saves images with our methode
    '''
    def segmentAndSave():
        DATA='D:/HAKIM/MIV M2/PFE/project/data/'+BDD+'/images/'
        DEST='D:/HAKIM/MIV M2/PFE/project/data/'+BDD+'/segmentation/'
        files=Data.loadFilesAsArray(DATA)
        for file in files:
            sImg=DATA+file
            img=cv2.imread(sImg,cv2.IMREAD_COLOR)
            # get contours
            contour=Contours.contours2(img)
            # draw contours
            img=np.zeros(np.shape(img))
            cv2.drawContours(img, [contour], -1, (255, 255, 255), -1)
            cv2.imwrite(DEST+file,img)
            # cv2.imshow('rt',img)
            # if cv2.waitKey() == ord('c'):
            #     break
        # cv2.destroyAllWindows()

    '''
        compares ground truth segmentation with our segmentation
        gets ratio between the area of imgTruth and the intersection
    '''
    @staticmethod
    def compare(img,imgTruth):
        intersection = cv2.bitwise_and(img, imgTruth)
        area = np.count_nonzero(img==255)
        areaTruth = np.count_nonzero(imgTruth==255)
        intersectionArea = np.count_nonzero(intersection==255)
        # return (intersectionArea/area + intersectionArea/areaTruth) / 2
        return 2*intersectionArea/(area+areaTruth)
    
    '''
        compares all the images for a giver database
    '''
    def compareAll():
        DATA_TRUTH='D:/HAKIM/MIV M2/PFE/project/data/'+BDD+'/truth/'
        DATA_SEGMENTATION='D:/HAKIM/MIV M2/PFE/project/data/'+BDD+'/segmentation/'
        files=Data.loadFilesAsArray(DATA_SEGMENTATION)
        # result = [np.array(['imgName','ratio'])]
        result=[]
        for file in files:
            img=cv2.imread(DATA_SEGMENTATION+file,cv2.IMREAD_GRAYSCALE)
            suffixe= '_lesion.bmp'
            if BDD=='ISIC':
                suffixe='.bmp'
            imgTruth=cv2.imread(DATA_TRUTH+file.replace('.bmp',suffixe),cv2.IMREAD_GRAYSCALE)
            ratio = Statistics.compare(img,imgTruth)
            result.append(np.array([file,ratio]))
        result=np.array(result)
        # print(result)
        result=result[:,1].astype(np.float)
        mean=np.mean(result)
        print(BDD,mean)
        # np.savetxt('Ratio results.txt',result)
        return mean

'''
    Statistics program
'''
BDD='PH2'
Statistics.segmentAndSave()
Statistics.compareAll()
BDD='ISIC'
Statistics.segmentAndSave()
Statistics.compareAll()