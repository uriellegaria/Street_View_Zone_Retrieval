import math 
import numpy as np
import sys
import matplotlib.pyplot as plt
from .GeoJSONHandler import GeoJSONOpener
import osmnx as ox
from tqdm import tqdm
from enum import Enum
import unicodedata


class StreetProperty(Enum):
    GVI_PROPERTY = "GVI"
    SVF_PROPERTY = "SVF"
    INFORMAL_RETAIL_PROPERTY = "Seller_Counts"
    BUILDING_COUNTS_PROPERTY = "Building_Counts"
    BUILDING_HEIGHT_PROPERTY = "Building_Height"
    SVI_COUNTS = "Counts"
    MOBILE_DATA_COUNTS = "Mobile_Counts"
    INFORMAL_RETAIL_FRACTION_PROPERTY = "Seller_fraction_occupation"

class PointDistanceCalculator: 

    def getDistance(self, latitude1, longitude1, latitude2, longitude2):
        return ox.distance.great_circle(latitude1, longitude1, latitude2, longitude2)

class Line:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    def __eq__(self, other):
        return self.point1[0] == other.point1[0] and self.point1[1] == other.point1[1] and self.point2[0] == other.point2[0] and self.point2[1] == other.point2[1]

    def getDistanceToPoint(self, point):

        point1 = np.array(self.point1)
        point2 = np.array(self.point2)
        pointArr = np.array(point)

        deltaVec = point2 - point1
        deltaVec2 = pointArr - point1

        dotProduct = np.dot(deltaVec, deltaVec2)
        normDelta = np.linalg.norm(deltaVec)

        alpha = dotProduct/(normDelta**2)

        if(alpha <= 1 and alpha >= 0):
            projectionPoint = point1 + alpha*(point2 - point1)
        
        else:
            projectionPoint = (point1 + point2)/2

        distanceSegment = [point[0] - projectionPoint[0], point[1] - projectionPoint[1]]

        distance = np.sqrt(distanceSegment[0]**2 + distanceSegment[1]**2)
        return distance

    def getPointProjection(self, point):
        point1 = np.array(self.point1)
        point2 = np.array(self.point2)
        pointArr = np.array(point)

        deltaVec = point2 - point1
        deltaVec2 = pointArr - point1

        dotProduct = np.dot(deltaVec, deltaVec2)
        normDelta = np.linalg.norm(deltaVec)

        alpha = dotProduct/(normDelta**2)

        projectionPoint = point1 + alpha*(point2 - point1)

        return projectionPoint
        
        

class Street:
    #I don't know how useful this class will be but i will include it.

    def __init__(self, streetId):
        self.streetId = streetId
        self.streetSegments = []
        self.segmentLengths = []
        #Dictionary that can be used to 
        self.attributes = {}
    
    def __eq__(self, other):
        return self.streetId == other.streetId
    
    def addSegment(self, streetSegmentPoints, segmentLength):
        if(not streetSegmentPoints in self.streetSegments and not streetSegmentPoints[::-1] in self.streetSegments):
            self.streetSegments.append(streetSegmentPoints)
            #We will actually compute the segmentLength again
            realLength = 0
            distanceCalculator = PointDistanceCalculator()
            for i in range(0,len(streetSegmentPoints) - 1):
                point1 = streetSegmentPoints[i]
                point2 = streetSegmentPoints[i+1]
                realLength = realLength + distanceCalculator.getDistance(point1[1], point1[0], point2[1], point2[0])
            
            self.segmentLengths.append(realLength)
    
    def getDistanceToPoint(self, point):
        lines = self.getAllLines()
        minDistIndex = -1
        minDist = float('inf')
        for i in range(0,len(lines)):
            dst = lines[i].getDistanceToPoint(point)
            if(dst < minDist):
                minDist = dst
                minDistIndex = i

        return minDist


    def getClosestLine(self, point):
        lines = self.getAllLines()
        minDistIndex = -1
        minDist = float('inf')
        for i in range(0,len(lines)):
            dst = lines[i].getDistanceToPoint(point)
            if(dst < minDist):
                minDist = dst
                minDistIndex = i

        return lines[minDistIndex]

    
    def getBearings(self):
        from Source.Utils.Utilities import GeometryCalculations

        calculator = GeometryCalculations()
        formatedPoints = self.getGoogleFormattedSamplingPoints()
        bearings = []
        deltaFraction = 0.1

        for i in range(len(formatedPoints)):
            point = np.array([formatedPoints[i][1], formatedPoints[i][0]])  
            closestLine = self.getClosestLine(point)
            point1 = np.array(closestLine.point1)
            point2 = np.array(closestLine.point2)

            offsetVector = (point2 - point1) * deltaFraction
            refPoint = point + offsetVector

            bearing = calculator.getBearing(point1[1], point1[0], point2[1], point2[0])
            bearings.append(bearing)

        return bearings


    def projectPointIntoStreet(self, point):
        lines = self.getAllLines()
        minDistIndex = float('nan')
        minDist = float('inf')
        for i in range(0,len(lines)):
            dst = lines[i].getDistanceToPoint(point)
            if(dst < minDist):
                minDist = dst
                minDistIndex = i
        
        projectedPoint = lines[minDistIndex].getPointProjection(point)

        return projectedPoint
    

    def sampleNIntermediatePoints(self, nPoints):
        streetDistance = self.getCompleteLength()/(nPoints+1)

        dst = 0
        points = []
        for i in range(0,nPoints):
            dst = dst + streetDistance
            point = self.getPointAtDistance(dst)
            if(not point is None):
                points.append(point)
        
        return points

    def getPointAtDistance(self, dst):
        cumulativeDst = 0
        segment = None
        #Search segment
        for i in range(0,len(self.segmentLengths)):
            if(dst >= cumulativeDst and dst <= cumulativeDst + self.segmentLengths[i]):
                segment = self.streetSegments[i]
                break
            else:
                cumulativeDst = cumulativeDst + self.segmentLengths[i]
        
        excessDistance = dst - cumulativeDst


        #Search subsegments
        subSegment = None
        nSegmentPoints = len(segment)
        distanceCalculator = PointDistanceCalculator()
        cumulativeDst = 0
        for i in range(0,nSegmentPoints - 1):
            point1 = segment[i]
            point2 = segment[i+1]
            distancePoints = distanceCalculator.getDistance(point1[1], point1[0], point2[1], point2[0])
            if(excessDistance >= cumulativeDst and excessDistance <= cumulativeDst + distancePoints):
                subSegment = [point1, point2]
                break
            else:
                cumulativeDst = cumulativeDst + distancePoints
        
        #Search a point that is closer to the distance
        alphaValues = np.linspace(0,1,100)
        vector1 = np.array(subSegment[0])
        vector2 = np.array(subSegment[1])

        for i in range(0,len(alphaValues)):
            alpha = alphaValues[i]
            vector = vector1 + alpha*(vector2 - vector1)
            distanceTo1 = distanceCalculator.getDistance(vector[1], vector[0], vector1[1], vector1[0])
            if(distanceTo1 - excessDistance > 0):
                return [vector[0], vector[1]]
        

        return [vector[0], vector[1]]

        

    def getAllLines(self):
        lines = []
        for i in range(0,len(self.streetSegments)):
            nSegmentPoints = len(self.streetSegments[i])
            for j in range(0,nSegmentPoints-1):
                point = self.streetSegments[i][j]
                nextPoint = self.streetSegments[i][j+1]
                line = Line(point, nextPoint)
                if(not line in lines):
                    lines.append(line)

        return lines

    def getCompleteLength(self):
        return sum(self.segmentLengths)

    def getPointsList(self):
        points = []
        for i in range(0,len(self.streetSegments)):
            nSegmentPoints = len(self.streetSegments[i])
            for j in range(0,nSegmentPoints):
                point = self.streetSegments[i][j]
                if(not point in points):
                    points.append(point)

        return points

    def getNumberOfPoints(self):
        return len(self.getPointsList())
    
    def getNumberOfSamplingPoints(self):
        return len(self.samplingPoints)
    

    def samplingPointExists(self, point):
        for i in range(0,len(self.samplingPoints)):
            samplingPoint = self.samplingPoints[i]
            if(samplingPoint[0] == point[0] and samplingPoint[1] == point[1]):
                return True
        
        return False
    
    def addSamplingPoint(self, point):
        if(not hasattr(self, 'samplingPoints')):
            self.samplingPoints = []
        
        if(not self.samplingPointExists(point)):
            self.samplingPoints.append(point)

    def setSamplingPoints(self, samplingPoints):
        self.samplingPoints = samplingPoints

    def getGoogleFormattedSamplingPoints(self):
        points = []
        for i in range(0,len(self.samplingPoints)):
            samplingPoint = self.samplingPoints[i]
            tuple = (samplingPoint[1], samplingPoint[0])
            points.append(tuple)

        return points
    

    def getGooglePointsAndBearings(self):
        googlePoints = self.getGoogleFormattedSamplingPoints()
        bearings = self.getBearings()

        return googlePoints, bearings
        
        
        

    def setAttributeValue(self, attributeName, attributeValue):
        self.attributes[attributeName] = attributeValue
    

    def getAttributeValue(self, attributeName):
        return self.attributes.get(attributeName, None)

    
        
        

class StreetSampler:
    
    
    def __init__(self, maxPoints):
        self.maxPoints = maxPoints
        self.streets = []
        self.minPointsPerStreet = 1
        self.unknownCounter = 0
        
    def getStreet(self, streetName, contains = False):
        if(not contains):
            streetName = unicodedata.normalize("NFC", streetName)
            for i in range(0,len(self.streets)):
                normalizedId = self.streets[i].streetId
                normalizedId = unicodedata.normalize("NFC", normalizedId)
                if(self.streets[i].streetId == streetName):
                    return self.streets[i]

            return None
        
        else:
            streetName = unicodedata.normalize("NFC", streetName)
            foundStreet = None
            for i in range(0,len(self.streets)):
                normalizedId = self.streets[i].streetId
                normalizedId = unicodedata.normalize("NFC", normalizedId)
                if(self.streets[i].streetId == streetName):
                    foundStreet = self.streets[i]
                    break
            
            if(foundStreet is None):
                for i in range(0,len(self.streets)):
                    normalizedId = self.streets[i].streetId
                    normalizedId = unicodedata.normalize("NFC", normalizedId)
                    if(self.streets[i].streetId in streetName or streetName in self.streets[i].streetId):
                        foundStreet = self.streets[i]
                        break


            return foundStreet
    
    def getNSamplingPoints(self):
        return len(self.getAllSamplingPoints())

    def printAttributes(self, attributeName):
        for i in range(0,len(self.streets)):
            print(self.streets[i].streetId+": "+str(self.streets[i].attributes[attributeName]))
    
    def euclideanDistance(self, point):
        return np.sqrt(point[0]**2 + point[1]**2)
    

    def printStreetNames(self):
        nStreets = len(self.streets)
        for i in range(0,nStreets):
            street = self.streets[i]
            print(street.streetId)

    def sampleStreetsNoIntersections(self):
        '''
        Only works for small areas
        '''
        #The weight of the street is the length
        #Larger streets receive more sampling points

        weights = []
        availablePoints = []
        for i in range(0,len(self.streets)):
            street = self.streets[i]
            weights.append(street.getCompleteLength())
            availablePoints.append(street.getNumberOfPoints())

        weights = np.array(weights)/sum(weights)

        for i in range(0,len(self.streets)):
            sampledPoints = []
            nSamplePoints = int(weights[i]*self.maxPoints)
            if(nSamplePoints < self.minPointsPerStreet):
                nSamplePoints = self.minPointsPerStreet

            #We need to collect one more extreme point than the actual number of points. 
            sampledPoints = self.streets[i].sampleNIntermediatePoints(nSamplePoints)

            self.streets[i].setSamplingPoints(sampledPoints)

    def printSampling(self):
        #Only call this after the points have been sampled
        for i in range(0,len(self.streets)):
            name = self.streets[i].streetId
            nSampling = len(self.streets[i].samplingPoints)

            print(name + ": "+str(nSampling)+" Points")

    def unknownStreetHasPoints(self, points):
        for i in range(0,len(self.streets)):
            name = self.streets[i].streetId
            if("unknown" in name and (points in self.streets[i].streetSegments or points[::-1] in self.streets[i].streetSegments)):
                return self.streets[i]

        return None

    def getAllSamplingPoints(self):
        samplingPoints = []
        for i in range(0,len(self.streets)):
            if(hasattr(self.streets[i], "samplingPoints")):
                samplingPoints.extend(self.streets[i].samplingPoints)
        return samplingPoints
    

    def getStreetOfNthPoint(self, n):
        samplingPointCount = 0
        currentStreet = None
        subIndex = None
        for i in range(0,len(self.streets)):
            if(hasattr(self.streets[i], "samplingPoints")):
                samplingPointCount = samplingPointCount + len(self.streets[i].samplingPoints)
                if(samplingPointCount > n):
                    currentStreet = self.streets[i]
                    subIndex = samplingPointCount % n
                    break
        return currentStreet, subIndex
    
    def drawSamplingScheme(self, width, height, pointColor = "#ff1764", nodeSize = "2", edgeSize = "2", edgeColor = "#ff700a"):
        plt.figure(figsize = (width, height))
        #Draw the streets
        nStreets = len(self.streets)
        for i in range(0,nStreets):
            streetLines = self.streets[i].getAllLines()
            for j in range(0,len(streetLines)):
                line = streetLines[j]
                plt.plot([line.point1[0], line.point2[0]], [line.point1[1], line.point2[1]], marker = "none", color = edgeColor, linewidth = edgeSize)
            
            if(hasattr(self.streets[i], "samplingPoints")):
                samplingPoints = self.streets[i].samplingPoints
                for j in range(0,len(samplingPoints)):
                    samplingPoint = samplingPoints[j]
                    plt.plot(samplingPoint[0], samplingPoint[1], marker = "o", color = pointColor, markersize = nodeSize)
        
        plt.axis("equal")


    

    def sampleWithProjectedLocations(self, locations, qualities = [], qualityThreshold = 0.8):
        '''
        Samples the streets using given locations. You can use the parameter minDst to reduce the amount of locations
        by choosing only places that are sufficiently separated between them. 

        Qualities indicates a number from 0 to 1, indicating the quality of each candidate point. If qualities is given they can be used
        along with the parameter qualityThreshold to limit the number of images that will be taken in the analyzed zone. The number of quality
        measurements needs to match the number of locations provided. 
        '''

        nLocations = len(locations)
        streets = []
        projectedPoints = []


        for i in range(0,nLocations):
            location = locations[i]

            #Let's obtain the closest street and project it. 
            projectedPoint,minStreet = self.projectPoint(location)
            #print(projectedPoint)
            streets.append(minStreet)
            projectedPoints.append(projectedPoint)
        

        if(len(qualities) != 0 and len(qualities) == len(locations)):
            for i in range(0,len(qualities)):
                quality = qualities[i]
                if(quality > qualityThreshold):
                    streets[i].addSamplingPoint(projectedPoints[i])


        else:
            #If qualities are not specified we will consider all projected points
            for i in range(0,len(streets)):
                streets[i].addSamplingPoint(projectedPoints[i])
    

    def restrictDistanceBetweenSamplingPoints(self, minDist = 2):
        nStreets = len(self.streets)
        distanceCalculator = PointDistanceCalculator()
        for i in tqdm(range(0,nStreets)):
            street = self.streets[i]
            if(hasattr(street, "samplingPoints")):
                samplingPoints = street.samplingPoints.copy()
                sortedPoints = sorted(samplingPoints, key=lambda x: x[0])
                sortedPoints = sorted(sortedPoints, key = lambda x: x[1])
                separationAchieved = False
                while(not separationAchieved and len(sortedPoints)>1):
                    for j in range(0,len(sortedPoints)-1):
                        point1 = sortedPoints[j]
                        point2 = sortedPoints[j+1]
                        dst = distanceCalculator.getDistance(point1[1], point1[0], point2[1], point2[0])
                        if(dst < minDist):
                            sortedPoints.pop(j+1)
                            break

                        if(j == len(sortedPoints) - 2):
                            separationAchieved = True

                
                street.samplingPoints = sortedPoints






    def projectPoint(self, point):
        minStreet = None
        minDst = float('inf')

        for i in range(0,len(self.streets)):
            street = self.streets[i]
            dst = street.getDistanceToPoint(point)
            
            if(dst < minDst):
                minDst = dst
                minStreet = street
            
        projectedPoint = minStreet.projectPointIntoStreet(point)
        return projectedPoint, minStreet

    

    def openStreetsWithDataFrame(self, dataFrame):
        gdf = dataFrame
        #The gdf must contain a lot of multi-line strings
        #I will assume each of them corresponds to one street
        #print(gdf.head(10))

        geomElements = list(gdf['geometry'])
        streetNames = list(gdf['name'])
        segmentLengths = list(gdf['length'])
        nSegments = len(geomElements)
        unknownCounter = 0
        unnamed = False
        
        for i in range(0,nSegments):
            segment = geomElements[i]
            streetName = streetNames[i]
            if(streetName == None or (type(streetName) == float and math.isnan(streetName))):
                streetName = "unnamed_"+str(self.unknownCounter)
                unnamed = True
            else:
                unnamed = False

            if(unnamed == False):
                if(type(streetName) == list):
                    streetName = streetName[0]
                street = Street(streetName)
                segmentLength = segmentLengths[i]
            
                if(not street in self.streets):
                    self.streets.append(street)
                else:
                    street = self.getStreet(streetName)
            
                nPoints = len(segment.coords)
                segmentPoints = []
                for j in range(0,nPoints):
                    pointTuple = segment.coords[j]
                    #I like arrays better than tuples
                    segmentPoints.append([pointTuple[0], pointTuple[1]])
            
                street.addSegment(segmentPoints, segmentLength)

            else:
                nPoints = len(segment.coords)
                segmentPoints = []
                for j in range(0,nPoints):
                    pointTuple = segment.coords[j]
                    segmentPoints.append([pointTuple[0], pointTuple[1]])
                
                possibleStreet = self.unknownStreetHasPoints(segmentPoints)
                if(possibleStreet == None):
                    street = Street(streetName)
                    self.streets.append(street)
                    street.addSegment(segmentPoints, segmentLength)
                    self.unknownCounter = self.unknownCounter + 1
    
    def tagStreets(self, attributeName, values):
        if(len(values) != len(self.streets)):
            print("Mismatch in number of values")
            return None
        
        nStreets = len(self.streets)
        for i in range(0,nStreets):
            street = self.streets[i]
            street.setAttributeValue(attributeName, values[i])
        
        print("Tagging finished")
                
    
    def openStreets(self, geoJSONPath):
        
        jsonHandler = GeoJSONOpener(geoJSONPath)
        self.geoJSONPath = geoJSONPath
        gdf = jsonHandler.getGeoDataFrame()
        #The gdf must contain a lot of multi-line strings
        #I will assume each of them corresponds to one street
        #print(gdf.head(10))
        
        geomElements = gdf['geometry']
        streetNames = list(gdf['name'])
        segmentLengths = list(gdf['length'])
        nSegments = len(geomElements)
        unnamed = False
        
        for i in range(0,nSegments):
            segment = geomElements[i]
            streetName = streetNames[i]
            if(streetName == None or (type(streetName) == float and math.isnan(streetName))):
                streetName = "unnamed_"+str(self.unknownCounter)
                unnamed = True
            else:
                unnamed = False

            if(unnamed == False):
                if(type(streetName) == list):
                    streetName = streetName[0]
                street = Street(streetName)
                segmentLength = segmentLengths[i]
            
                if(not street in self.streets):
                    self.streets.append(street)
                else:
                    street = self.getStreet(streetName)
            
                nPoints = len(segment.coords)
                segmentPoints = []
                for j in range(0,nPoints):
                    pointTuple = segment.coords[j]
                    #I like arrays better than tuples
                    segmentPoints.append([pointTuple[0], pointTuple[1]])
            
                street.addSegment(segmentPoints, segmentLength)

            else:
                nPoints = len(segment.coords)
                segmentPoints = []
                for j in range(0,nPoints):
                    pointTuple = segment.coords[j]
                    segmentPoints.append([pointTuple[0], pointTuple[1]])
                
                possibleStreet = self.unknownStreetHasPoints(segmentPoints)
                if(possibleStreet == None):
                    street = Street(streetName)
                    self.streets.append(street)
                    segmentLength = segmentLengths[i]
                    street.addSegment(segmentPoints, segmentLength)
                    self.unknownCounter = self.unknownCounter + 1
                    
                    


                






