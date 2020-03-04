import os

class Data:
    @staticmethod
    def loadFilesAsArray(folder):
        files=os.listdir(folder)
        return files