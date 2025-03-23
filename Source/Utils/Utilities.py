import random
import random
import numpy as np
import math
import shutil
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from enum import Enum
import csv



class ColorsWalkabilityVars(Enum):


    GVI_MIN_COLOR = [148/255.0, 255/255.0, 176/255.0]
    GVI_MAX_COLOR = [5/255.0, 59/255.0, 19/255.0] 
    SVF_MIN_COLOR = [150/255.0, 206/255.0, 255/255.0]
    SVF_MAX_COLOR = [4/255.0, 36/255.0, 64/255.0]
    BUILDING_HEIGHTS_MIN_COLOR = [232/255, 230/255, 230/255]
    BUILDING_HEIGHTS_MAX_COLOR = [46/255.0, 45/255.0, 45/255.0]
    COUNTS_MIN_COLOR = [255/255.0, 186/255.0, 186/255.0]
    COUNTS_MAX_COLOR = [125/255.0, 29/255.0, 29/255.0]
    INFORMAL_RETAIL_MIN_COLOR = [255/255.0, 200/255.0, 179/255.0]
    INFORMAL_RETAIL_MAX_COLOR = [240/255.0, 96/255.0, 41/255.0]
    




class GeometryCalculations:
    '''
    Class that encapsulates some special geometrical calculations
    '''
    def __init__(self):
        self.earthRadius = 6378.14 *10**3

    def getAngleBetweenVectors(self, vector1, vector2):
        
        angleRadians = np.abs(np.arccos(np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))))
        angleDegrees = angleRadians*(180/np.pi)
        return angleDegrees

    def pointInsideBox(self, point, xMin, xMax, yMin, yMax):
        xCoord = point[0]
        yCoord = point[1]

        if(xCoord >= xMin and xCoord <= xMax and yCoord >= yMin and yCoord <= yMax):
            return True
        else:
            return False
    
    def getDistanceBetweenCoordinates(self, latitude1, longitude1, latitude2, longitude2):
        #Earth radius
        R = 6371
        latitude1Radians = math.radians(latitude1)
        longitude1Radians = math.radians(longitude1)
        latitude2Radians = math.radians(latitude2)
        longitude2Radians = math.radians(longitude2)
        
        dst = 2*R*math.asin(np.sqrt((1 - np.cos(latitude2Radians - latitude1Radians) + np.cos(latitude1Radians)*np.cos(latitude2Radians)*(1 - np.cos(longitude2Radians - longitude1Radians)))/2))

        return dst*10**3
    
    def getBearing(self, latitude1, longitude1, latitude2, longitude2):
        lat1 = math.radians(latitude1)
        lat2 = math.radians(latitude2)

        diffLong = math.radians(longitude2 - longitude1)

        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)* math.cos(lat2)*math.cos(diffLong))

        initialBearing = math.atan2(x, y)


        initialBearing = math.degrees(initialBearing)
        compassBearing = (initialBearing + 360) % 360

        return compassBearing
    
    

class FileCopier:

    def copyFile(self, originPath, destinationPath):
        shutil.copy(originPath, destinationPath)


class ColorMapCreator:

    def getColorMap(self, colorArray1, colorArray2, minValue, maxValue):
        '''
        Receives the extremes of a color gradient and returns the colormap associated with the scale
        so that it can be placed in an image, i will actually return the norm object too. 
        '''
        cdict = {
            'red':(
                (0,colorArray1[0], colorArray1[0]), 
                (1, colorArray2[0], colorArray2[0])

            ),

            'green':(
                (0, colorArray1[1], colorArray1[1]),
                (1, colorArray2[1], colorArray2[1])
            ),

            'blue':(
                (0, colorArray1[2], colorArray1[2]),
                (1, colorArray2[2], colorArray2[2])
            )

        }

        cmapName = "custom_cmap"
        nBins = 100
        cmap = LinearSegmentedColormap(cmapName, cdict, N = 100)
        norm = mpl.colors.Normalize(vmin = minValue, vmax = maxValue)

        return norm, cmap 


class CSVManager:

    def exportCSV(self, headers, rows, path):
        with open(path, 'w') as csvfile:
            write = csv.writer(csvfile)
            write.writerow(headers)
            write.writerows(rows)
    
    def readCSV(self, filePath):
        with open(filePath, mode='r') as file:
            csvReader = csv.reader(file)
            headers = next(csvReader)
            data = [row for row in csvReader]
        return headers, data


class RandomColorGenerator:

    def getRandomColor(self):
        redValue = random.random()
        greenValue = random.random()
        blueValue = random.random()

        return np.array([redValue, greenValue, blueValue])

    def rgb2Hex(self, color):
        red = int(color[0]*255)
        blue = int(color[1]*255)
        green = int(color[2]*255)
        
        redString = "0x{:02x}".format(red)[2:]
        blueString = "0x{:02x}".format(blue)[2:]
        greenString = "0x{:02x}".format(green)[2:]

        hexString = "#"+redString+blueString+greenString

        return hexString
    

    def hex2RGB(self, hexColor):
        hexColor = hexColor.lstrip('#')
        red = int(hexColor[0:2], 16)/255.0
        green = int(hexColor[2:4], 16)/255.0
        blue = int(hexColor[4:6], 16)/255.0

        return np.array([red, green, blue])

    def getHexColor(self):
        return self.rgb2Hex(self.getRandomColor())

    def getRandomColors(self, nColors, separation = 0.1):

        colors = []
        maxIter = 10*nColors
        iters = 0
        while(len(colors) < nColors and iters < maxIter):
            newColor = self.getRandomColor()
            addColor = True
            for i in range(0,len(colors)):
                color = colors[i]
                dst = np.sqrt((newColor[0] - color[0])**2 + (newColor[1] - color[1])**2 + (newColor[2] - color[2])**2)
                if(dst < separation):
                    addColor = False

            if(addColor):
                colors.append(newColor)

            iters = iters + 1

        if(iters >= maxIter):
            print("Iterations exceeded error")
            return -1
        else:

            hexColors = []
            for i in range(0,len(colors)):
                hexValue = self.rgb2Hex(colors[i])
                hexColors.append(hexValue)
            return hexColors

