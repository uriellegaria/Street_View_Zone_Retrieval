import geopandas as gpd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class GeoJSONOpener:


    def __init__(self, path):
        '''
        Takes as argument the path where the geojson file is located. 
        '''
        self.path = path
        #Visualization parameters
        self.nodeSize = 1
        self.edgeSize = 1

    def setNodeSize(self, nodeSize):
        self.nodeSize = nodeSize

    def setEdgeSize(self, edgeSize):
        self.edgeSize = edgeSize

    def getGeoDataFrame(self):
        '''
        Returns a geodataframe from the GeoJSON file. 
        '''
        #Open the GDF 
        gdf = gpd.read_file(self.path)
        return gdf

    def getBBOX(self, graph):
        '''
        Returns a bbox indicating [(minX, minY), (maxX, minY), (maxX, maxY), (minX, maxY)]
        '''

        graph = self.getVisualizationGraph()
        minX = float('inf')
        maxX = float('-inf')
        minY = float('inf')
        maxY = float('-inf')

        nodeKeys = list(graph.nodes.keys())

        for i in range(0,len(nodeKeys)):
            nodeKey = nodeKeys[i]
            nodePosition = graph.nodes[nodeKey]['pos']
            nodeX = nodePosition[0]
            nodeY = nodePosition[1]
            if(nodeX < minX):
                minX = nodeX
            if(nodeX > maxX):
                maxX = nodeX
            if(nodeY > maxY):
                maxY = nodeY
            if(nodeY < minY):
                minY = nodeY

        return np.array([[minX, minY], [maxX, minY], [maxX, maxY], [minX, maxY]])
        

    def getVisualizationGraph(self):
        '''
        Returns a networkx graph of the dataframe. This function omits non-geometrical information that could be included in the geodataframe
        and we reserve its use for visualization and debugging purposes.
        '''
        graph = nx.Graph()
        geoDataFrame = self.getGeoDataFrame()

        #Add nodes and connections 
        geometryGDF = geoDataFrame['geometry']
        nElements = len(geometryGDF)

        for i in range(0,len(geometryGDF)):
            geometricObject = geometryGDF[i]
            nCoords = len(geometricObject.coords)
            for j in range(0,nCoords):
                coordinates = geometricObject.coords[j]
                graph.add_node(coordinates, pos = coordinates)
                if(j > 0):
                    #I am quite sure this distance should have to be
                    #transformed somehow, but for now i will use an euclidean
                    #metric
                    previousCoords = geometricObject.coords[j-1]
                    dst = np.sqrt((previousCoords[0] - coordinates[0])**2 + (previousCoords[1] - coordinates[1])**2)
                    graph.add_edge(previousCoords, coordinates, length = dst)
                    


        return graph


    def drawMap(self, ax, graph, nodeColor = "#2ac1db", edgeColor = "#e38d3d", drawNodes = False, drawEdges = True):
        '''
        Draws the nodes and/or the edges of a graph
        '''
        
        nodeKeys = list(graph.nodes.keys())
        nNodes = len(nodeKeys)

        edgeKeys = list(graph.edges.keys())
        nEdges = len(edgeKeys)
        
        #First draw the edges
        if(drawEdges):
            for i in range(0,nEdges):
                edgeKey = edgeKeys[i]
                fromPosition = edgeKey[0]
                toPosition = edgeKey[1]
                ax.plot([fromPosition[0], toPosition[0]], [fromPosition[1], toPosition[1]], color = edgeColor, linewidth = self.edgeSize, marker = "none")
        
        #Draw the nodes
        if(drawNodes):
            for i in range(0,nNodes):
                nodeKey = nodeKeys[i]
                nodePosition = graph.nodes[nodeKey]['pos']
                #Plot the node
                ax.plot(nodePosition[0], nodePosition[1], color = nodeColor, marker = "o", markersize = self.nodeSize)

        ax.set_axis_off()
    


    def drawMapInstant(self, width = 7, height = 7, nodeColor = "#2ac1db", edgeColor = "#e38d3d", drawNodes = False, drawEdges = True):
        graph = self.getVisualizationGraph()
        fig, ax = plt.subplots(figsize = (width, height))
        self.drawMap(ax, graph, nodeColor = nodeColor, edgeColor = edgeColor, drawNodes = drawNodes, drawEdges = drawEdges)