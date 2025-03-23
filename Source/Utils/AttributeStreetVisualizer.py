#Let's create a class that can be used to achieve a visualization of a map based on attribute 
#variables.

#3 types of variables

from enum import Enum
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
from .Utilities import ColorMapCreator
import matplotlib as mpl

class VariableType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1
    CATEGORICAL = 2

#Ok so the path to get to using this class would be Having a geojson -> Using the StreetSampleTools to open strets
#-> Adding attributes to the streets (for instance with some street view analysis)-> Visualizing the streets

class StreetAttributesVisualizer:

    def __init__(self, streetSampler):
        self.streetSampler = streetSampler


    def colorByAttribute(self, attributeName, attributeType, width, height, edgeSize, *args, orientationBar="horizontal", showAxis=False, minValue=None, maxValue=None):
        # For continuous variables args contains two triplets: 1. the RGB for min value color
        # and 2. The RGB for max value.

        # For discrete variables (counts of people, obstacles, etc.), you could actually use the
        # same color scheme as continuous variables. But i will provide an option to specify the 
        # value of the variable on each street. args here provides the font size, which has to be 
        # adapted to the size of the map.

        # Finally, for categorical variables args provides a dictionary specifying the color corresponding
        # to each value. Example red = sidewalk, blue = no sidewalk. Or red = more than 5 obstacles, blue = less than
        # five obstacles
        if(attributeType == VariableType.CONTINUOUS):
            minValueColor = args[0]
            maxValueColor = args[1]

            # We first need to get the value of the attribute for all streets
            attributeValues = []
            streets = self.streetSampler.streets
            for i in range(0, len(streets)):
                street = streets[i]
                attributeValue = street.attributes[attributeName]
                attributeValues.append(attributeValue)

            # Use provided minValue and maxValue if given; otherwise, compute from data
            if minValue is None:
                minValue = np.min(attributeValues)
            if maxValue is None:
                maxValue = np.max(attributeValues)

            for i in range(0, len(streets)):
                street = streets[i]
                attributeValue = attributeValues[i]
                if(maxValue - minValue > 0):
                    normalizedValue = (attributeValue - minValue) / (maxValue - minValue)
                    clippedValue = np.clip(normalizedValue, 0, 1)
                    color = minValueColor + (maxValueColor - minValueColor) * clippedValue
                    street.setAttributeValue("color", color)
                else:
                    color = minValueColor
                    street.setAttributeValue("color", color)

            # Now that we have been assigned a value we can draw the streets
            fig, ax = plt.subplots(figsize=(width, height), layout='constrained')
            colorMapCreator = ColorMapCreator()
            norm, colorMap = colorMapCreator.getColorMap(minValueColor, maxValueColor, minValue, maxValue)

            for i in range(0, len(streets)):
                street = streets[i]
                streetColor = street.attributes["color"]
                segments = street.streetSegments
                for j in range(0, len(segments)):
                    segment = segments[j]
                    xCoords = [x[0] for x in segment]
                    yCoords = [x[1] for x in segment]
                    plt.plot(xCoords, yCoords, linewidth=edgeSize, color=streetColor)

            ax.axis("equal")
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colorMap), ax=ax, orientation=orientationBar, label=attributeName)
            if not showAxis:
                ax.set_axis_off()
            
        elif(attributeType == VariableType.DISCRETE):
            minValueColor = args[0]
            maxValueColor = args[1]
            fontSize = args[2]
            displacement = args[3]
            
            #We first need to get the value of the attribute for all streeets

            attributeValues = []
            streets = self.streetSampler.streets
            for i in range(0,len(streets)):
                street = streets[i]
                attributeValue = street.attributes[attributeName]
                attributeValues.append(attributeValue)


            minValue = np.min(attributeValues)
            maxValue = np.max(attributeValues)

            for i in range(0,len(streets)):
                street = streets[i]
                attributeValue = attributeValues[i]
                color = minValueColor + (maxValueColor - minValueColor)*((attributeValue-minValue)/(maxValue - minValue))
                street.setAttributeValue("color", color)


            #Now that we have been asigned a value we can draw the streets
            fig, ax = plt.subplots(figsize = (width, height), layout = 'constrained')
            mapCreator = ColorMapCreator()
            norm, colorMap = mapCreator.getColorMap(minValueColor, maxValueColor, minValue, maxValue)
            
            for i in range(0,len(streets)):
                street = streets[i]
                streetColor = street.attributes["color"]
                segments = street.streetSegments
                attributeValue = attributeValues[i]
                for j in range(0,len(segments)):
                    segment = segments[j]
                    xCoords = [x[0] for x in segment]
                    yCoords = [x[1] for x in segment]
                    ax.plot(xCoords, yCoords, linewidth = edgeSize, color = streetColor)

                
                points = street.getPointsList()
                xCoords = [x[0] for x in points]
                yCoords = [x[1] for x in points]

                #Get the centroid
                xAverage = sum(xCoords)/len(xCoords)
                yAverage = sum(yCoords)/len(yCoords)

                centroid = [xAverage, yAverage]
                projectedCentroid = self.getProjectedCentroid(centroid, xCoords, yCoords)

                xAverage = projectedCentroid[0]
                yAverage = projectedCentroid[1]

                distances = [((x - xAverage) ** 2 + (y - yAverage) ** 2) ** 0.5 for x, y in zip(xCoords, yCoords)]
                sortedIndices = sorted(range(len(distances)), key=lambda i: distances[i])
                closestPointsIndices = sortedIndices[:2]

                point1 = [xCoords[closestPointsIndices[0]], yCoords[closestPointsIndices[0]]]
                point2 = [xCoords[closestPointsIndices[1]], yCoords[closestPointsIndices[1]]]

                directionVector = [point2[0] - point1[0], point2[1] - point1[1]]

                directionVectorNorm = (directionVector[0]**2 + directionVector[1]**2)**0.5

                directionVector[0] = directionVector[0]/directionVectorNorm
                directionVector[1] = directionVector[1]/directionVectorNorm

                metersDisplacement = displacement
                latitude = yAverage  
                metersPerDegreeLat = 111320  
                metersPerDegreeLon = 111320 * np.cos(np.radians(latitude))
                latDisplacement = metersDisplacement / metersPerDegreeLat
                lonDisplacement = metersDisplacement / metersPerDegreeLon

                displacedX = xAverage + lonDisplacement*directionVector[0]
                displacedY = yAverage + latDisplacement*directionVector[1]

                minSize = int(fontSize/3)
                fontScaledSize = int(minSize + (attributeValue/maxValue)*(fontSize - minSize))
                #ax.scatter(displacedX, displacedY, color='red', s=20)
                ax.text(displacedX, displacedY, int(attributeValue), fontsize = fontScaledSize, ha = "center", va = "center")

            #Let's create a dummy mappable
            fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = colorMap), ax = ax, orientation = orientationBar, label = attributeName)
            if(not showAxis):
                ax.set_axis_off()

        elif(attributeType == VariableType.CATEGORICAL):
            colorDictionary = args[0]
            #In this case we can directly start drawing the streets with the appropriate color
            streets = self.streetSampler.streets
            fig, ax = plt.subplots(figsize = (width, height), layout = 'constrained')

            colorPatches = []
            colorDictionaryKeys = list(colorDictionary.keys())
            for i in range(0,len(colorDictionaryKeys)):
                key = colorDictionaryKeys[i]
                colorAttribute = colorDictionary[key]
                keyString = str(key)
                patch = patches.Patch(facecolor = colorAttribute, label = keyString)
                colorPatches.append(patch)
                
            for i in range(0,len(streets)):
                street = streets[i]
                attributeValue = street.attributes[attributeName]
                streetColor = colorDictionary[attributeValue]
                segments = street.streetSegments
                for j in range(0,len(segments)):
                    segment = segments[j]
                    xCoords = [x[0] for x in segment]
                    yCoords = [x[1] for x in segment]
                    plt.plot(xCoords, yCoords, color = streetColor, linewidth = edgeSize)
            
            plt.legend(handles=colorPatches, loc="best")


            if(not showAxis):
                plt.gca().set_axis_off()
    

    def projectPointOntoSegment(self,point, segmentStart, segmentEnd):
        segmentVector = np.array([segmentEnd[0] - segmentStart[0], segmentEnd[1] - segmentStart[1]])
    
        pointVector = np.array([point[0] - segmentStart[0], point[1] - segmentStart[1]])

        segmentLengthSquared = segmentVector[0] ** 2 + segmentVector[1] ** 2
    
        if segmentLengthSquared == 0:
            return segmentStart

        t = np.dot(pointVector, segmentVector) / segmentLengthSquared

        t = max(0, min(1, t))
    
        projectedPoint = [segmentStart[0] + t * segmentVector[0], segmentStart[1] + t * segmentVector[1]]
    
        return projectedPoint
    

    def getRecommendedExtremeValues(self, data, minPercentile=0.05, maxPercentile=0.95):
        data = np.array(data)
        nData = len(data)
        nBins = int(np.sqrt(nData))  

        distribution = np.zeros(nBins)
        m = np.min(data)
        M = np.max(data)

        binWidth = (M - m) / nBins

        for dataPoint in data:
            binNumber = int((dataPoint - m) / (M - m) * nBins)  
            if binNumber == nBins:  
                binNumber -= 1
            distribution[binNumber] += 1

        distribution = distribution / sum(distribution)

        cumulativeDistribution = np.zeros(nBins)
        cumulativeSum = 0
        for i in range(nBins):
            cumulativeSum += distribution[i]
            cumulativeDistribution[i] = cumulativeSum

        minRecommendation = None
        maxRecommendation = None
        for i in range(nBins):
            if minRecommendation is None and cumulativeDistribution[i] >= minPercentile:
                minRecommendation = m + i * binWidth  
            if maxRecommendation is None and cumulativeDistribution[i] >= maxPercentile:
                maxRecommendation = m + i * binWidth
                break 

        return minRecommendation, maxRecommendation

    def getProjectedCentroid(self, centroid, xCoords, yCoords):

    # Calculate the total length of the street
        totalLength = 0
        segmentLengths = []
        for i in range(len(xCoords) - 1):
            segmentLength = np.sqrt((xCoords[i + 1] - xCoords[i]) ** 2 + (yCoords[i + 1] - yCoords[i]) ** 2)
            segmentLengths.append(segmentLength)
            totalLength += segmentLength

    # Find the midpoint in terms of distance
        midpoint = totalLength / 2

    # Find the segment that contains the midpoint
        cumulativeLength = 0
        for i in range(len(segmentLengths)):
            cumulativeLength += segmentLengths[i]
            if cumulativeLength >= midpoint:
            # We found the segment that contains the midpoint
                segmentStart = [xCoords[i], yCoords[i]]
                segmentEnd = [xCoords[i + 1], yCoords[i + 1]]
            # Project the centroid onto this segment
                projectedPoint = self.projectPointOntoSegment(centroid, segmentStart, segmentEnd)
                return projectedPoint

    # Fallback: if no projection is found (should not happen), return the centroid
        return centroid
                    

            


