import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import imageio
import os

import cv2
import skimage.morphology

sys.path.append("/Users/jiayinlu/Desktop/Jiayin/projects/wing/sknw-master")
import sknw

import networkx as nx

import copy

from cellpose import plot, utils
from cellpose import models
from cellpose.io import imread

import shapely.geometry# import Polygon

from matplotlib.collections import PatchCollection
import matplotlib.patches #import Polygon

population_id=152



#Read files
file_path_outline ="/Users/jiayinlu/Downloads/images_stacks_wings_extraction/segmentation/population_{}/outline/".format(population_id)
file_list_outline = []
# Iterate directory
for file in os.listdir(file_path_outline):
    # check only text files
    if file.endswith('outline.npy'):
        file_list_outline.append(file)

sorted_file_list_outline=sorted(file_list_outline)

num_wings=len(sorted_file_list_outline)



#Read files
file_path_cellpose ="/Users/jiayinlu/Downloads/images_stacks_wings_extraction/segmentation/population_{}/cellpose/".format(population_id)
file_list_cellpose = []
# Iterate directory
for file in os.listdir(file_path_cellpose):
    # check only text files
    if file.endswith('seg.npy'):
        file_list_cellpose.append(file)

sorted_file_list_cellpose=sorted(file_list_cellpose)

#output files paths
file_path_cell="/Users/jiayinlu/Downloads/images_stacks_wings_extraction/segmentation/population_{}/cell_analysis/".format(population_id)
file_path_vein="/Users/jiayinlu/Downloads/images_stacks_wings_extraction/segmentation/population_{}/vein_analysis/".format(population_id)

#A. Wing area
#Input: categorized cell-vein-background image
#Output: area of wing (pixel)
population_wings_area=[]
for population_id in [1,60,152]:
    #Read files
    file_path_outline ="/Users/jiayinlu/Downloads/images_stacks_wings_extraction/segmentation/population_{}/outline/".format(population_id)
    file_list_outline = []
    # Iterate directory
    for file in os.listdir(file_path_outline):
        # check only text files
        if file.endswith('outline.npy'):
            file_list_outline.append(file)

    sorted_file_list_outline=sorted(file_list_outline)

    num_wings=len(sorted_file_list_outline)

    #Read files
    file_path_cellpose ="/Users/jiayinlu/Downloads/images_stacks_wings_extraction/segmentation/population_{}/cellpose/".format(population_id)
    file_list_cellpose = []
    # Iterate directory
    for file in os.listdir(file_path_cellpose):
        # check only text files
        if file.endswith('seg.npy'):
            file_list_cellpose.append(file)

    sorted_file_list_cellpose=sorted(file_list_cellpose)
    
    wings_area=[]
    x_temp=[]
    for i in range(0,num_wings):
        vein_cell_bg=np.load(file_path_outline + sorted_file_list_outline[i])
        wing_area=np.count_nonzero(vein_cell_bg !=0.5)
        wings_area.append(wing_area)
        x_temp.append(population_id)
    wings_area=np.array(wings_area)
    population_wings_area.append(wings_area)

    if population_id==1:
        plt.plot(x_temp,wings_area,"g*",label="population {}".format(population_id))
    if population_id==60:
        plt.plot(x_temp,wings_area,"r*",label="population {}".format(population_id))
    if population_id==152:
        plt.plot(x_temp,wings_area,"b*",label="population {}".format(population_id))
plt.ylabel("wing area")
plt.xlabel("population")
plt.legend()





#B. cell fractional area, cell circularity
#Input: cellpose output
#Output: area of each cell
for population_id in [1, 60,152]:
    #Read files
    file_path_outline ="/Users/jiayinlu/Downloads/images_stacks_wings_extraction/segmentation/population_{}/outline/".format(population_id)
    file_list_outline = []
    # Iterate directory
    for file in os.listdir(file_path_outline):
        # check only text files
        if file.endswith('outline.npy'):
            file_list_outline.append(file)

    sorted_file_list_outline=sorted(file_list_outline)

    num_wings=len(sorted_file_list_outline)

    #Read files
    file_path_cellpose ="/Users/jiayinlu/Downloads/images_stacks_wings_extraction/segmentation/population_{}/cellpose/".format(population_id)
    file_list_cellpose = []
    # Iterate directory
    for file in os.listdir(file_path_cellpose):
        # check only text files
        if file.endswith('seg.npy'):
            file_list_cellpose.append(file)

    sorted_file_list_cellpose=sorted(file_list_cellpose)
    
    
    wings_area=[]
    x_temp=[]
    for i in range(0,num_wings):
        vein_cell_bg=np.load(file_path_outline + sorted_file_list_outline[i])
        wing_area=np.count_nonzero(vein_cell_bg !=0.5)
        wings_area.append(wing_area)
    wings_area=np.array(wings_area)
    
    for i in range(0,1):#num_wings):
        wing_cellpose=np.load(file_path_cellpose + sorted_file_list_cellpose[i],allow_pickle=True).item()
        cell_contours=utils.outlines_list(wing_cellpose['masks'])
        
        cells_polygon=[]
        cells_area=[]
        cells_perimeter=[]
        for o in cell_contours:
            coords=[]
            for ii in range(0,len(o)):
                coords.append(o[ii])
            coords.append(o[0])
            #smooth polygon boundary to avoid raster fractal length effect
            cells_polygon.append(shapely.geometry.Polygon(coords).simplify(tolerance=0.8))
            cells_area.append(cells_polygon[-1].area)
            cells_perimeter.append(cells_polygon[-1].length)
        cells_area=np.array(cells_area)
        cells_perimeter=np.array(cells_perimeter)
        cells_fractional_area=cells_area/wings_area[i]
        cells_circularity=4*np.pi*cells_area/(cells_perimeter**2)
        
        if i==0 and population_id==60:
            plt.plot(cells_fractional_area,cells_circularity,"r*",label="population {}".format(population_id))
        elif i==0 and population_id==152:
            plt.plot(cells_fractional_area,cells_circularity,"b*",label="population {}".format(population_id))
        elif i==0 and population_id==1:
            plt.plot(cells_fractional_area,cells_circularity,"g*",label="population {}".format(population_id))
        elif population_id==152:
            plt.plot(cells_fractional_area,cells_circularity,"b*")
        elif population_id==1:
            plt.plot(cells_fractional_area,cells_circularity,"g*")
        else:
            plt.plot(cells_fractional_area,cells_circularity,"r*")
    plt.ylabel("cell circularity")
    plt.xlabel("cell fractional area")
    plt.title("one indiviidual from each population")
    plt.legend()






for i in range(0,num_wings):
    wing_cellpose=np.load(file_path_cellpose + sorted_file_list_cellpose[i],allow_pickle=True).item()
    cell_contours=utils.outlines_list(wing_cellpose['masks'])
    
    cells_polygon=[]
    cells_area=[]
    cells_perimeter=[]
    for o in cell_contours:
        coords=[]
        for ii in range(0,len(o)):
            coords.append(o[ii])
        coords.append(o[0])
        #smooth polygon boundary to avoid raster fractal length effect
        cells_polygon.append(shapely.geometry.Polygon(coords).simplify(tolerance=0.8))
        cells_area.append(cells_polygon[-1].area)
        cells_perimeter.append(cells_polygon[-1].length)
    cells_area=np.array(cells_area)
    cells_perimeter=np.array(cells_perimeter)
    cells_fractional_area=cells_area/wings_area[i]
    cells_circularity=4*np.pi*cells_area/(cells_perimeter**2)
    
   
    #plt.plot(*cells_polygon[0].exterior.xy,"*-")
    #plt.plot(*cells_polygon[0].simplify(tolerance=0.8).exterior.xy,"*-")
    
    #B1. figure of cell fractional areas
    fig,ax = plt.subplots(1,figsize=(40, 30)) 
    N = len(cell_contours)
   
    patches = []
    
    norm = matplotlib.colors.Normalize(vmin=min(cells_fractional_area), vmax=max(cells_fractional_area))
    cmap = plt.get_cmap('viridis')
    colors = cmap(norm(cells_fractional_area))
    
    for o in cell_contours:
        coords=[]
        for ii in range(0,len(o)):
            coords.append(o[ii])
        coords.append(o[0])
        patches.append(matplotlib.patches.Polygon(coords,closed=True))
        
    
    collection = PatchCollection(patches)
    
    im=ax.add_collection(collection)
    
    collection.set_color(colors)
    
    ax.autoscale_view()
    
    plt.axis('off')

    cbar=fig.colorbar(im, orientation='vertical')
    cbar.ax.tick_params(labelsize=40)
    plt.title("{}\ncell fractional areas\n[0,1]:[{},{}]".format(sorted_file_list_cellpose[i][:-10],min(cells_fractional_area),max(cells_fractional_area)),fontsize=50)
    plt.savefig(file_path_cell+sorted_file_list_cellpose[i][:-10]+"_hw_cell_fractional_areas.png")
    plt.close()


    
    #B2. figure of cell circularity/compactness:[0,1], 1 for a circle
    #Polsby-Popper test:https://en.wikipedia.org/wiki/Polsby–Popper_test
    fig,ax = plt.subplots(1,figsize=(40, 30)) 
    N = len(cell_contours)
   
    patches = []
    
    
    norm = matplotlib.colors.Normalize(vmin=min(cells_circularity), vmax=max(cells_circularity))
    cmap = plt.get_cmap('viridis')
    colors = cmap(norm(cells_circularity))
    
    for o in cell_contours:
        coords=[]
        for ii in range(0,len(o)):
            coords.append(o[ii])
        coords.append(o[0])
        patches.append(matplotlib.patches.Polygon(coords,closed=True))
        
    
    collection = PatchCollection(patches)
    
    im=ax.add_collection(collection)
    
    collection.set_color(colors)
    
    ax.autoscale_view()
    
    plt.axis('off')

    cbar=fig.colorbar(im, orientation='vertical')
    cbar.ax.tick_params(labelsize=40)
    plt.title("{}\ncell circularity/compactness\nmin,max:{},{}".format(sorted_file_list_cellpose[i][:-10],min(cells_circularity),max(cells_circularity)),fontsize=50)
    plt.savefig(file_path_cell+sorted_file_list_cellpose[i][:-10]+"_hw_cell_circularity.png")
    plt.close()













#C. Vein network


for population_id in [1, 60,152]:
    #Read files
    file_path_outline ="/Users/jiayinlu/Downloads/images_stacks_wings_extraction/segmentation/population_{}/outline/".format(population_id)
    file_list_outline = []
    # Iterate directory
    for file in os.listdir(file_path_outline):
        # check only text files
        if file.endswith('outline.npy'):
            file_list_outline.append(file)

    sorted_file_list_outline=sorted(file_list_outline)

    num_wings=len(sorted_file_list_outline)

    #Read files
    file_path_cellpose ="/Users/jiayinlu/Downloads/images_stacks_wings_extraction/segmentation/population_{}/cellpose/".format(population_id)
    file_list_cellpose = []
    # Iterate directory
    for file in os.listdir(file_path_cellpose):
        # check only text files
        if file.endswith('seg.npy'):
            file_list_cellpose.append(file)

    sorted_file_list_cellpose=sorted(file_list_cellpose)
    
    
    wings_area=[]
    x_temp=[]
    for i in range(0,num_wings):
        vein_cell_bg=np.load(file_path_outline + sorted_file_list_outline[i])
        wing_area=np.count_nonzero(vein_cell_bg !=0.5)
        wings_area.append(wing_area)
    wings_area=np.array(wings_area)
    
    
    for i in range(0,num_wings):
        
        internal_length_sum=0
        perimeter_scaled=0
        
        print(file_path_outline+sorted_file_list_outline[i])
        gray=np.load(file_path_outline + sorted_file_list_outline[i]) #0.5: background, 0: cell, vein: 1
        
        binary=gray.copy()
        
        nnx=len(binary[0,:])
        ny=len(binary[:,0])
        
        for jj in range(0,ny):
            for ii in range(0,nnx):
                if binary[jj,ii]==0.5:
                    binary[jj,ii]=0
                else:
                    binary[jj,ii]=1
        
        
        ret,thresh = cv2.threshold(binary,0.5,1,0)
        thresh = thresh.astype(np.uint8)
        contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        
        
        # B. Extract Vein Network skeleton and estimate thickness from Binary Vein data
        thresh=binary.astype('uint8')
        
        # get distance transform
        distance = thresh.copy()
        distance = cv2.distanceTransform(distance, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)
        
        
        
        # get skeleton (medial axis)
        binary2 = thresh.copy()
        skeleton = skimage.morphology.skeletonize(binary2).astype(np.float32)
        
        
        # apply skeleton to select center line of distance 
        thickness = cv2.multiply(distance, skeleton)*2
        
        
        # get average thickness for non-zero pixels
        average = np.median(thickness[skeleton!=0])

        # thickness average
        thick =average
        
        
        #C. Extract vein network (nodes, edges) from vein skeleton 
        ske = skeleton.astype(np.uint16)
        
        # build graph from skeleton
        graph = sknw.build_sknw(ske,multi=False,ring=False, iso=False)
        
        
        #Remove disconnected component and keep only the largest connected graph
        N_subs=1
        G_sub=[]
        largest_components=[]
        for ii in range(N_subs):
          largest_components.append(sorted(nx.connected_components(graph), key=len, reverse=True)[ii])
          G_sub.append(graph.subgraph(largest_components[ii]))
        
        G_subgraphs=nx.compose_all(G_sub)
        #draw_vein_network(G_subgraphs)

        #Associate thickness and length attributes to each edge
        edge_thickness_list=[]
        edge_length_list=[]
        for (s,e) in G_subgraphs.edges():
            ps = G_subgraphs[s][e]['pts']
            avg_edge_thickness=0.0
            ei_npt=len(ps)
            for ei in range(0,ei_npt):
                eix=ps[ei][1]
                eiy=ps[ei][0]
                avg_edge_thickness=avg_edge_thickness+thickness[eiy,eix]
            avg_edge_thickness=avg_edge_thickness/ei_npt
            
            G_subgraphs[s][e]['weight'] = avg_edge_thickness
            G_subgraphs[s][e]['length'] = ei_npt
            
            edge_thickness_list.append(G_subgraphs[s][e]['weight'])
            edge_length_list.append(G_subgraphs[s][e]['length'])
            
        internal_length_sum=np.sum(np.array(edge_length_list))/perimeter
        perimeter_scaled=perimeter/area*area
        
        
        plt.plot(perimeter_scaled,internal_length_sum,"*")
            
        if i==0 and population_id==60:
            plt.plot(perimeter_scaled,internal_length_sum,"r*",label="population {}".format(population_id))
        elif i==0 and population_id==152:
            plt.plot(perimeter_scaled,internal_length_sum,"b*",label="population {}".format(population_id))
        elif i==0 and population_id==1:
            plt.plot(perimeter_scaled,internal_length_sum,"g*",label="population {}".format(population_id))
        elif population_id==152:
            plt.plot(perimeter_scaled,internal_length_sum,"b*")
        elif population_id==1:
            plt.plot(perimeter_scaled,internal_length_sum,"g*")
        else:
            plt.plot(perimeter_scaled,internal_length_sum,"r*")
    plt.ylabel("sum of internal length(scaled)")
    plt.xlabel("perimeter(scaled)")
    plt.title("indiviiduals from populations")
    plt.legend()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    for i in range(0,1):#num_wings):
        wing_cellpose=np.load(file_path_cellpose + sorted_file_list_cellpose[i],allow_pickle=True).item()
        cell_contours=utils.outlines_list(wing_cellpose['masks'])
        
        cells_polygon=[]
        cells_area=[]
        cells_perimeter=[]
        for o in cell_contours:
            coords=[]
            for ii in range(0,len(o)):
                coords.append(o[ii])
            coords.append(o[0])
            #smooth polygon boundary to avoid raster fractal length effect
            cells_polygon.append(shapely.geometry.Polygon(coords).simplify(tolerance=0.8))
            cells_area.append(cells_polygon[-1].area)
            cells_perimeter.append(cells_polygon[-1].length)
        cells_area=np.array(cells_area)
        cells_perimeter=np.array(cells_perimeter)
        cells_fractional_area=cells_area/wings_area[i]
        cells_circularity=4*np.pi*cells_area/(cells_perimeter**2)
        
        if i==0 and population_id==60:
            plt.plot(cells_fractional_area,cells_circularity,"r*",label="population {}".format(population_id))
        elif i==0 and population_id==152:
            plt.plot(cells_fractional_area,cells_circularity,"b*",label="population {}".format(population_id))
        elif i==0 and population_id==1:
            plt.plot(cells_fractional_area,cells_circularity,"g*",label="population {}".format(population_id))
        elif population_id==152:
            plt.plot(cells_fractional_area,cells_circularity,"b*")
        elif population_id==1:
            plt.plot(cells_fractional_area,cells_circularity,"g*")
        else:
            plt.plot(cells_fractional_area,cells_circularity,"r*")
    plt.ylabel("cell circularity")
    plt.xlabel("cell fractional area")
    plt.title("one indiviidual from each population")
    plt.legend()



def draw_vein_network(graph):
    plt.figure(figsize=(40,25))
    plt.gca().invert_yaxis()
    # draw edges by pts
    for (s,e) in graph.edges():
        ps = graph[s][e]['pts']
        plt.plot(ps[:,1], ps[:,0], 'green')
        
    # draw node by o
    nodes = graph.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])
    plt.scatter(ps[:,1], ps[:,0], s=70, c="r")
    
    
    # title and show
    plt.title('Build Graph')
    plt.show()
    
    
for i in range(0,num_wings):
    print(file_path_outline+sorted_file_list_outline[i])
    gray=np.load(file_path_outline + sorted_file_list_outline[i]) #0.5: background, 0: cell, vein: 1
    
    binary=gray.copy()
    
    nnx=len(binary[0,:])
    ny=len(binary[:,0])
    
    for jj in range(0,ny):
        for ii in range(0,nnx):
            if binary[jj,ii]==0.5:
                binary[jj,ii]=0
    
    # B. Extract Vein Network skeleton and estimate thickness from Binary Vein data
    thresh=binary.astype('uint8')
    
    # get distance transform
    distance = thresh.copy()
    distance = cv2.distanceTransform(distance, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)
    
    
    
    # get skeleton (medial axis)
    binary2 = thresh.copy()
    skeleton = skimage.morphology.skeletonize(binary2).astype(np.float32)
    
    
    # apply skeleton to select center line of distance 
    thickness = cv2.multiply(distance, skeleton)*2
    
    
    # get average thickness for non-zero pixels
    average = np.median(thickness[skeleton!=0])

    # thickness average
    thick =average
    
    
    #C. Extract vein network (nodes, edges) from vein skeleton 
    ske = skeleton.astype(np.uint16)
    
    # build graph from skeleton
    graph = sknw.build_sknw(ske,multi=False,ring=False, iso=False)
    
    
    #Remove disconnected component and keep only the largest connected graph
    N_subs=1
    G_sub=[]
    largest_components=[]
    for ii in range(N_subs):
      largest_components.append(sorted(nx.connected_components(graph), key=len, reverse=True)[ii])
      G_sub.append(graph.subgraph(largest_components[ii]))
    
    G_subgraphs=nx.compose_all(G_sub)
    #draw_vein_network(G_subgraphs)

    #Associate thickness and length attributes to each edge
    edge_thickness_list=[]
    edge_length_list=[]
    for (s,e) in G_subgraphs.edges():
        ps = G_subgraphs[s][e]['pts']
        avg_edge_thickness=0.0
        ei_npt=len(ps)
        for ei in range(0,ei_npt):
            eix=ps[ei][1]
            eiy=ps[ei][0]
            avg_edge_thickness=avg_edge_thickness+thickness[eiy,eix]
        avg_edge_thickness=avg_edge_thickness/ei_npt
        
        G_subgraphs[s][e]['weight'] = avg_edge_thickness
        G_subgraphs[s][e]['length'] = ei_npt
        
        edge_thickness_list.append(G_subgraphs[s][e]['weight'])
        edge_length_list.append(G_subgraphs[s][e]['length'])
        
        
    #Figure: thickness of veins
    fig,ax = plt.subplots(1,figsize=(40, 30)) 

    for (s,e) in G_subgraphs.edges():
        ps = G_subgraphs[s][e]['pts']
        ax.plot(ps[:,1], ps[:,0], c=cm.jet(G_subgraphs[s][e]['weight']/np.mean(edge_thickness_list)),linewidth=8)

    plt.axis('off')

    #plt.colorbar()
    #cbar.ax.tick_params(labelsize=40)
     
    plt.title("{}\nvein thickness".format(sorted_file_list_outline[i][:-12]),fontsize=50)
    plt.savefig(file_path_vein+sorted_file_list_outline[i][:-12]+"_vein_thickness.png")
    plt.close()

    
    
    #Figure: modularity communities
    c = nx.community.greedy_modularity_communities(G_subgraphs)
        
    list_integers = np.array(range(0,len(c)))
    #cmap=cm.rainbow(np.array(list_integers)/np.mean(list_integers))
        
    fig,ax = plt.subplots(1,figsize=(40, 30)) 
        
    # draw edges by pts
    for (s,e) in graph.edges():
        ps = graph[s][e]['pts']
        ax.plot(ps[:,1], ps[:,0], 'k')
        
    #draw nodes by community 
    nodes = graph.nodes()
    cict=0
    for ci in c:
        for ni in ci:
            ps = np.array([nodes[ni]['o']])
            ax.scatter(ps[:,1], ps[:,0], s=500, c=cm.rainbow(list_integers[cict]/np.mean(list_integers)))
        cict=cict+1
    
    ax.autoscale_view()
    
    plt.axis('off')
    plt.title("{}\nvenation topology\nmaximum modularity communities".format(sorted_file_list_outline[i][:-12]),fontsize=50)
    plt.savefig(file_path_vein+sorted_file_list_outline[i][:-12]+"_venation_topology.png")
    plt.close()
        
    
    
    
    
    
    
    
    
    
    
    
    




