import os
import sys
dirname = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(dirname, '../sknw-master')))
import sknw

import argparse

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import imageio

import cv2
import skimage.morphology

import networkx as nx

import copy

import cv2

from cellpose import plot, utils
from cellpose import models
from cellpose.io import imread

import shapely.geometry# import Polygon

from matplotlib.collections import PatchCollection
import matplotlib.patches #import Polygon
import matplotlib.ticker as mticker

import matplotlib as mpl


import re
import glob


import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import load_workbook
import math

from matplotlib.collections import LineCollection

import pickle


class wing_venation_network:
    
    #Set up paths to input and output data
    def __init__(self,population, species, base_dir, save_dir):
        
        # Specify input directory
        self.base_dir=base_dir
        # Specify directory to save results
        self.save_dir=save_dir
        #Specify this image population and species
        self.population=population
        self.species=species
        
         
        # ====================== Read in wing image
        self.img0=mpimg.imread(self.base_dir+'/analysis/segmentation/population_{}/svd_result//population_{}+FMNH_{}_hw_1.png'.format(self.population, self.population, self.species))    #test image
        self.ny0=len(self.img0[:, 0])  # y
        self.nx0=len(self.img0[0, :])  # x
    
        
        # ====================== Read in image that categorize vein 1, cell 0, background 0.5. 
        self.vein_cell_bg=np.load(self.base_dir+'/analysis/segmentation/population_{}/outline/population_{}+FMNH_{}_hw_outline.npy'.format(self.population, self.population, self.species))    #test image
        
        # ====================== Read in cellpose segmentation info
        pattern = os.path.join(
                                self.base_dir,
                                'analysis/segmentation/population_{}/cellpose/population_{}+FMNH_{}_hw_*seg.npy'.format(
                                    self.population, self.population, self.species
                                )
                            )
        # Search for matching file
        matches = glob.glob(pattern)
        
        self.wing_cellpose = np.load(matches[0], allow_pickle=True).item()
        self.cell_contours=utils.outlines_list(self.wing_cellpose['masks'])
        self.num_domains=len(self.cell_contours)
        
        
        #Some statistical variables to use later
        self.wing_area=0
        self.cells_polygon=[]
        self.cells_area=[]
        self.cells_perimeter=[]
        

    def calculate_wing_area(self):        
        self.wing_area=np.count_nonzero(self.vein_cell_bg !=0.5)
    
    def calculate_domain_stats(self):
        for o in self.cell_contours:
            if len(o)>0:
                coords=[]
                for ii in range(0,len(o)):
                    coords.append(o[ii])
                coords.append(o[0])
                #smooth polygon boundary to avoid raster fractal length effect
                self.cells_polygon.append(shapely.geometry.Polygon(coords).simplify(tolerance=0.8))
                self.cells_area.append(self.cells_polygon[-1].area)
                self.cells_perimeter.append(self.cells_polygon[-1].length)
        self.cells_area=np.array(self.cells_area)
        self.cells_perimeter=np.array(self.cells_perimeter)
        self.cells_fractional_area=self.cells_area/self.wing_area
        self.cells_circularity=4*np.pi*self.cells_area/(self.cells_perimeter**2)
        
    
    def plot_cell_areas(self):
        fig,ax = plt.subplots(1,figsize=(38,23)) 
        
        N = len(cell_contours)
        patches = []
        for o in cell_contours:
            if len(o)>0:
                coords=[]
                for ii in range(0,len(o)):
                    coords.append(o[ii])
                coords.append(o[0])
                patches.append(matplotlib.patches.Polygon(coords,closed=True))
                
        collection = PatchCollection(patches,cmap="inferno")
        collection.set_array(cells_fractional_area)
        collection.set_clim([0.0005,0.006])
        
        im=ax.add_collection(collection)
        ax.autoscale_view()
        plt.axis('off')
        cbar=fig.colorbar(im, orientation='vertical')
        cbar.ax.tick_params(labelsize=40)
        
        plt.gca().invert_yaxis()
        plt.title("{}\ncell fractional areas".format(sorted_file_list_cellpose[i][:-10]),fontsize=50)
        
        
        plt.tight_layout()
        plt.savefig(file_path_cell+sorted_file_list_cellpose[i][:-10]+"_hw_cell_fractional_areas.png")
        plt.close()
        
    
        
        
    
    
    
    
    
        
    
        
    
    
        
    
        #B2. figure of cell circularity/compactness:[0,1], 1 for a circle
        #Polsby-Popper test:https://en.wikipedia.org/wiki/Polsby–Popper_test
        fig,ax = plt.subplots(1,figsize=(38,23)) 
        
        N = len(cell_contours)
        patches = []
        for o in cell_contours:
            if len(o)>0:
                coords=[]
                for ii in range(0,len(o)):
                    coords.append(o[ii])
                coords.append(o[0])
                patches.append(matplotlib.patches.Polygon(coords,closed=True))
                
            
        collection = PatchCollection(patches,cmap="inferno")
        collection.set_array(cells_circularity)
        collection.set_clim([0.5,0.85])
        
        im=ax.add_collection(collection)    
        ax.autoscale_view()
        plt.axis('off')
        cbar=fig.colorbar(im, orientation='vertical')
        cbar.ax.tick_params(labelsize=40)
        
        plt.gca().invert_yaxis()
        plt.title("{}\ncell circularity".format(sorted_file_list_cellpose[i][:-10]),fontsize=50)
        plt.show()
    
        plt.savefig(file_path_cell+sorted_file_list_cellpose[i][:-10]+"_hw_cell_circularity.png")
        plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #C. Vein network
     
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
        fig,ax = plt.subplots(1,figsize=(38,23)) 
        segx=[]
        segy=[]
        clist=[]
        for (s,e) in G_subgraphs.edges():
            ps = G_subgraphs[s][e]['pts']
            segx.append(ps[:,1].tolist())
            segy.append(ps[:,0].tolist())
            clist.append(G_subgraphs[s][e]['weight'])
            
        segments = [np.column_stack([x, y]) for x, y in zip(segx, segy)]
        lc = LineCollection(segments,cmap="rainbow", lw=8)
        lc.set_array(clist)
        lc.set_clim([10,20])
    
        im=ax.add_collection(lc)
        ax.autoscale_view()
        plt.axis('off')
        cbar=fig.colorbar(im, orientation='vertical')
        cbar.ax.tick_params(labelsize=40)
    
    
        plt.gca().invert_yaxis()
        plt.title("{}\nvein thickness".format(sorted_file_list_outline[i][:-12]),fontsize=50)
        plt.show()
    
        plt.savefig(file_path_vein+sorted_file_list_outline[i][:-12]+"_vein_thickness.png")
        plt.close()
    
    
        
        
        #Figure: modularity communities
        c = nx.community.greedy_modularity_communities(G_subgraphs)
        list_integers = np.array(range(0,len(c)))        
        fig,ax = plt.subplots(1,figsize=(32,23)) 
        # draw edges by pts
        for (s,e) in G_subgraphs.edges():
            ps = G_subgraphs[s][e]['pts']
            ax.plot(ps[:,1], ps[:,0], 'k')
            
        #draw nodes by community 
        nodes = G_subgraphs.nodes()
        cict=0
        for ci in c:
            for ni in ci:
                ps = np.array([nodes[ni]['o']])
                ax.scatter(ps[:,1], ps[:,0], s=500, c=cm.gist_rainbow(list_integers[cict]/np.mean(list_integers)))
            cict=cict+1
        
        ax.autoscale_view()
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.title("{}\nvenation topology\nmaximum modularity communities".format(sorted_file_list_outline[i][:-12]),fontsize=50)
        plt.show()
        
        plt.savefig(file_path_vein+sorted_file_list_outline[i][:-12]+"_venation_topology.png")
        plt.close()
            
        
        # find boundary edges, nodes and save graph object to file
        fn_prefix=file_path_vein+sorted_file_list_outline[i][:-12]
        find_graph_boundary(G_subgraphs,fn_prefix)
        
        
###############################################
################Functions######################
###############################################
def draw_vein_network(graph):
    #plt.figure(figsize=(40,25))
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

def find_graph_boundary(G2,fn_prefix):
    nodes=G2.nodes()
    nx.set_node_attributes(G2, 0, "boundary")
    maxx=-1
    maxy=-1
    for ni in nodes:
        #coordinate of node ni, eg. array([  70, 2644], dtype=uint16)
        nodes[ni]['boundary']=0
        node_xy=nodes[ni]['o']
        if node_xy[1]>maxx:
            maxx=node_xy[1]
        if node_xy[0]>maxy:
            maxy=node_xy[0]
    NX=maxx+50
    NY=maxy+50
    background_temp=np.zeros((NY,NX))
    
    for ni in nodes:
        #coordinate of node ni, eg. array([  70, 2644], dtype=uint16)
        node_xy=nodes[ni]['o']
        
        #Incident edges to a node
        for (s,e) in G2.edges(ni): 
            edge_coords=G2[s][e]['pts']  
            for [y,x] in edge_coords:
                background_temp[y-1:y+1,x-1:x+1]=1


    #Apply flood-fill with seed (NY-1, NX-1) to obtain a background mask 
    background_temp=np.float32(background_temp)
    floodval = 0.5
    background_temp=cv2.floodFill(background_temp, None, (NX-10, NY-10), floodval)
    background_temp=background_temp[1]
      


    #Found out boundary edges and nodes, denote boundary nodes and edge attribute 'boundary'=1
    for ni in nodes:
        #Incident edges to a node
        for (s,e) in G2.edges(ni): 
            edge_coords=G2[s][e]['pts']  
            G2[s][e]['boundary']=0
            edge_mid_loc=int(len(edge_coords)/2)
            ptmy,ptmx=edge_coords[edge_mid_loc]
            
            ptiy, ptix =edge_coords[0]
            ptey, ptex =edge_coords[-1]
            perp_vec=np.asarray([ptiy-ptey,ptex-ptix])
            if np.abs(ptiy-ptey)<1:
                perp_vec=np.asarray([0,1])
            
            perp_vec=perp_vec/np.linalg.norm(perp_vec)
            perx,pery=perp_vec
            
            dis=min(max(G2[s][e]['length']/2,3),10)
            
            testy,testx=int(ptmy+dis*pery),int(ptmx+dis*perx)
            if background_temp[testy,testx]==0.5:
                G2[s][e]['boundary']=1
                nodes[ni]['boundary']=1
                
            else:
                testy,testx=int(ptmy-dis*pery),int(ptmx-dis*perx)
                if background_temp[testy,testx]==0.5:
                    G2[s][e]['boundary']=1
                    nodes[ni]['boundary']=1
    #Loop through edges again, make sure no bdry edge left un-accounted
    for ni in nodes:
        #Incident edges to a node
        for (s,e) in G2.edges(ni): 
            if nodes[s]['boundary']==1 and nodes[e]['boundary']==1:
                G2[s][e]['boundary']=1
    
    
    # save graph object to file
    #nx.write_gpickle(G2, fn_prefix+'_vein_graph.pickle')
    with open(fn_prefix+'_vein_graph.pickle', 'wb') as f:
        pickle.dump(G2, f, pickle.HIGHEST_PROTOCOL)

    

################################################
#################Inter-vein angle###############
################################################


    
i=0
# load graph object from file
#G2 = nx.read_gpickle(file_path_vein+sorted_file_list_outline[i][:-12]+'_vein_graph.pickle')
#nodes=G2.nodes()
      
with open(file_path_vein+sorted_file_list_outline[i][:-12]+'_vein_graph.pickle', 'rb') as f:
    G2 = pickle.load(f)
nodes=G2.nodes()
    
for ni in nodes:
    #Incident edges to a node
    for (s,e) in G2.edges(ni): 
        if G2[s][e]['boundary']==1:
            edge_coords=G2[s][e]['pts']  
            plt.plot(edge_coords[:,1],edge_coords[:,0], "r-")
            #plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="y")
plt.show()

#https://stackoverflow.com/questions/15191088/how-to-do-a-polynomial-fit-with-fixed-points
def polyfit_with_fixed_points(n, x, y, xf, yf) :
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x**np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf**np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = xf_n / 2
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)
    return params[:n + 1]
      
 
edge_length_list=[]
for ni in nodes:
    #Incident edges to a node
    for (s,e) in G2.edges(ni): 
        edge_length_list.append(G2[s][e]['length'] )
edge_length_list=np.asarray(edge_length_list)
median_len=np.median(edge_length_list)
search_angle_thres=2.0/5.0*median_len


all_inter_edge_angle=[]
nx.set_node_attributes(G2, 0, "golden_angle")
nx.set_node_attributes(G2, 0, "104_angle")
nx.set_node_attributes(G2, 0, "160_angle")

for ni in nodes:
    #coordinate of node ni, eg. array([  70, 2644], dtype=uint16)
    node_xy=nodes[ni]['o']
    
    #polar_deta list
    polar_deta_list=[]
    edge_e_list=[]
    
    #Incident edges to a node
    for (s,e) in G2.edges(ni): 
        edge_e_list.append(e)
        edge_coords=G2[s][e]['pts']  
        #correct orientation of edge so it starts from ni
        if (node_xy==edge_coords[-1]).all():
            edge_coords=edge_coords[::-1]
            
        #Trace through line segments on edge, until the distance from begin node ni to the segment end node ei is more than threshold RR
        #Edge length
        EL=len(edge_coords)
        #Threshold, 1/5 of edge length
        RR=int(1.0/3.0*EL) #min(min(int(search_angle_thres),int(0.5*EL)),EL)
        if RR==1: 
            RR=EL-1
        
        #fitX=edge_coords[:RR][:,1]
        #fitY=edge_coords[:RR][:,0]
        #coef = np.polyfit(fitX, fitY, 1)
        #poly1d_fn = np.poly1d(coef)
        
        
        #xnew=edge_coords[:RR][1,1]
        #ynew=poly1d_fn(xnew)
        
        #endxy=np.array((ynew,xnew))
        #origin_xy=np.array((poly1d_fn(edge_coords[:RR][0,1]),edge_coords[:RR][0,1]))
        
        
     
        endxy=edge_coords[RR]
        
        #Obtain vector of the direction of the edge
        edge_vec=endxy-node_xy
        #edge_vec=endxy-origin_xy
        #Find angle deta of polar coordinate
        deta=np.arctan2(edge_vec[1],edge_vec[0])
        if deta<0:
            deta=2*np.pi+deta
        deta=deta*180/np.pi
        #Collect edge polar angle to list
        polar_deta_list.append(deta)
        
    #Sort polar angles in ascending order, obtain corresponding INDEX
    sorted_IND=np.argsort(polar_deta_list)
    #Obtain the corresponding sorted angle list and edge endpoint list
    polar_deta_list=np.asarray(polar_deta_list)
    edge_e_list=np.asarray(edge_e_list)
    polar_deta_list_sorted=polar_deta_list[sorted_IND]
    edge_e_list_sorted=edge_e_list[sorted_IND]
    
    #inter-edge angles list
    inter_edge_angle_list=[]
    
    #Not a boundary node, normal procedure
    if nodes[ni]['boundary']!=1: 
        for ai in range(1,len(polar_deta_list)):
            big_angle=polar_deta_list_sorted[ai]
            small_angle=polar_deta_list_sorted[ai-1]
            inter_edge_angle_list.append(big_angle-small_angle)
        inter_edge_angle_list.append(polar_deta_list_sorted[0]+(360.0-polar_deta_list_sorted[-1]))
        
    #Boundary node, need to exclude angle between boundary edges
    else: 
        
        for ai in range(1,len(polar_deta_list_sorted)):
            #Connecting end-points of the corresponding edges
            e0_temp=edge_e_list_sorted[ai-1]
            e1_temp=edge_e_list_sorted[ai]
            
            #Check if the angle is in-between two boundary edges
            #If not, store the inter-edge angle
            if nodes[e0_temp]['boundary']!=1 or nodes[e1_temp]['boundary']!=1:
                big_angle=polar_deta_list_sorted[ai]
                small_angle=polar_deta_list_sorted[ai-1]
                inter_edge_angle_list.append(big_angle-small_angle)
        
        e0_temp=edge_e_list_sorted[-1]
        e1_temp=edge_e_list_sorted[0]
        #Check if the angle is in-between two boundary edges
        #If not, store the inter-edge angle
        if nodes[e0_temp]['boundary']!=1 or nodes[e1_temp]['boundary']!=1:
            inter_edge_angle_list.append(polar_deta_list_sorted[0]+(360.0-polar_deta_list_sorted[-1]))
               
    all_inter_edge_angle.extend(inter_edge_angle_list)
    for ai in inter_edge_angle_list:
        if ai<137+5 and ai>137-5:
            nodes[ni]['golden_angle']=1
        if ai<165 and ai>155:
            nodes[ni]['160_angle']=1
        if ai<109 and ai>99:
            nodes[ni]['104_angle']=1
        
            
all_inter_edge_angle=np.asarray(all_inter_edge_angle)
        



        
plt.figure(figsize=(40,25))
for ni in nodes:
    #Incident edges to a node
    for (s,e) in G2.edges(ni): 
        edge_coords=G2[s][e]['pts']  
        plt.plot(edge_coords[:,1],edge_coords[:,0], "k-")
        #plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="y")
        if G2[s][e]['boundary']==1:
            plt.plot(edge_coords[:,1],edge_coords[:,0], "r-")
            #plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="y")
        #if nodes[ni]['large_angle']==1:
        #    plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=50, c="b")
        if nodes[ni]['golden_angle']==1:
            plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=400, c="b")
            
plt.show()

plt.figure(figsize=(40,25))
for ni in nodes:
    #Incident edges to a node
    for (s,e) in G2.edges(ni): 
        edge_coords=G2[s][e]['pts']  
        plt.plot(edge_coords[:,1],edge_coords[:,0], "k-")
        #plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="y")
        if G2[s][e]['boundary']==1:
            plt.plot(edge_coords[:,1],edge_coords[:,0], "r-")
            #plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="y")
        #if nodes[ni]['large_angle']==1:
        #    plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=50, c="b")
        if nodes[ni]['golden_angle']==1:
            plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="r")
        if nodes[ni]['95_angle']==1:
            plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="b")
        if nodes[ni]['165_angle']==1:
            plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="y")
        
plt.show()

plt.figure(figsize=(40,25))
for ni in nodes:
    #Incident edges to a node
    for (s,e) in G2.edges(ni): 
        edge_coords=G2[s][e]['pts']  
        plt.plot(edge_coords[:,1],edge_coords[:,0], "k-")
        #plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="y")
        if G2[s][e]['boundary']==1:
            plt.plot(edge_coords[:,1],edge_coords[:,0], "r-")
            #plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="y")
        #if nodes[ni]['large_angle']==1:
        #    plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=50, c="b")
        #if nodes[ni]['golden_angle']==1:
        #    plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="r")
        if nodes[ni]['104_angle']==1:
            plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="r")
        
        
plt.show()



plt.figure(figsize=(40,30))
plt.scatter(0,0, s=300, c="y",label="104")
plt.scatter(0,0, s=600, c="r",label="160")
for ni in nodes:
    #Incident edges to a node
    for (s,e) in G2.edges(ni): 
        edge_coords=G2[s][e]['pts']  
        plt.plot(edge_coords[:,1],edge_coords[:,0], "k-")
        #plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="y")
        if G2[s][e]['boundary']==1:
            plt.plot(edge_coords[:,1],edge_coords[:,0], "r-")
            #plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="y")
        #if nodes[ni]['large_angle']==1:
        #    plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=50, c="b")
        #if nodes[ni]['golden_angle']==1:
        #    plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="r")
        if nodes[ni]['160_angle']==1:
            plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=600, c="r")
        if nodes[ni]['104_angle']==1:
            plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=300, c="y")
        
plt.axis("off")
plt.legend(fontsize=40)
plt.xlim((5,3500))
plt.ylim((5,3000))

plt.show()


plt.figure(figsize=(40,30))
plt.scatter(0,0, s=400, c="y",label="104+/-5")
for ni in nodes:
    #Incident edges to a node
    for (s,e) in G2.edges(ni): 
        edge_coords=G2[s][e]['pts']  
        plt.plot(edge_coords[:,1],edge_coords[:,0], "k-")
        #plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="y")
        if G2[s][e]['boundary']==1:
            plt.plot(edge_coords[:,1],edge_coords[:,0], "r-")
            #plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="y")
        #if nodes[ni]['large_angle']==1:
        #    plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=50, c="b")
        #if nodes[ni]['golden_angle']==1:
        #    plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="r")
        if nodes[ni]['104_angle']==1:
            plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=400, c="y")
        
plt.axis("off")
plt.legend(fontsize=40)
plt.xlim((5,3500))
plt.ylim((5,3000))

plt.show()
      

plt.figure(figsize=(40,30))
plt.scatter(0,0, s=400, c="r",label="160+/-5")
for ni in nodes:
    #Incident edges to a node
    for (s,e) in G2.edges(ni): 
        edge_coords=G2[s][e]['pts']  
        plt.plot(edge_coords[:,1],edge_coords[:,0], "k-")
        #plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="y")
        if G2[s][e]['boundary']==1:
            plt.plot(edge_coords[:,1],edge_coords[:,0], "r-")
            #plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="y")
        #if nodes[ni]['large_angle']==1:
        #    plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=50, c="b")
        #if nodes[ni]['golden_angle']==1:
        #    plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="r")
        if nodes[ni]['160_angle']==1:
            plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=400, c="r")
        
plt.axis("off")
plt.legend(fontsize=40)
plt.xlim((5,3500))
plt.ylim((5,3000))

plt.show()


plt.figure(figsize=(40,30))
plt.scatter(0,0, s=400, c="b",label="golden 137+/-5")
for ni in nodes:
    #Incident edges to a node
    for (s,e) in G2.edges(ni): 
        edge_coords=G2[s][e]['pts']  
        plt.plot(edge_coords[:,1],edge_coords[:,0], "k-")
        #plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="y")
        if G2[s][e]['boundary']==1:
            plt.plot(edge_coords[:,1],edge_coords[:,0], "r-")
            #plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="y")
        #if nodes[ni]['large_angle']==1:
        #    plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=50, c="b")
        #if nodes[ni]['golden_angle']==1:
        #    plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=100, c="r")
        if nodes[ni]['golden_angle']==1:
            plt.scatter(nodes[ni]['o'][1],nodes[ni]['o'][0], s=400, c="b")
        
plt.axis("off")
plt.legend(fontsize=40)
plt.xlim((5,3500))
plt.ylim((5,3000))

plt.show()
      

        
np.min(all_inter_edge_angle)     
np.max(all_inter_edge_angle)           
        
        
        
            
all_inter_edge_angle=all_inter_edge_angle[all_inter_edge_angle<355]

#https://stackoverflow.com/questions/55187037/how-can-i-do-a-histogram-with-1d-gaussian-mixture-with-sklearn
from sklearn.mixture import GaussianMixture as GMM 
from scipy.stats import norm
X=np.asarray(all_inter_edge_angle).reshape(-1, 1)  


# first of all, let's confirm the optimal number of components
bics = []
min_bic = 0
counter=1
for i in range (5): # test the AIC/BIC metric between 1 and 10 components
  gmm = GMM(n_components = counter, max_iter=1000, random_state=0, covariance_type = 'full')
  labels = gmm.fit(X).predict(X)
  bic = gmm.bic(X)
  bics.append(bic)
  if bic < min_bic or min_bic == 0:
    min_bic = bic
    opt_bic = counter
  counter = counter + 1


# plot the evolution of BIC/AIC with the number of components
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1,2,1)
# Plot 1
plt.plot(np.arange(1,6), bics, 'o-', lw=3, c='black', label='BIC')
plt.legend(frameon=False, fontsize=15)
plt.xlabel('Number of components', fontsize=20)
plt.ylabel('Information criterion', fontsize=20)
plt.xticks(np.arange(0,11, 2))
plt.title('Opt. components = '+str(opt_bic), fontsize=20)
plt.show()

# Since the optimal value is n=4 according to both BIC and AIC, let's write down:
n_optimal = 2 #opt_bic

# create GMM model object
gmm = GMM(n_components = n_optimal, max_iter=1000, random_state=10, covariance_type = 'full').fit(X)

# find useful parameters
mean = gmm.means_  
covs  = gmm.covariances_
weights = gmm.weights_ 


plt.figure(figsize=(40,25))
# create necessary things to plot
x_axis = np.arange(0,300, 1)
y_axis0 = norm.pdf(x_axis, float(mean[0][0]), np.sqrt(float(covs[0][0][0])))*weights[0] # 1st gaussian
y_axis1 = norm.pdf(x_axis, float(mean[1][0]), np.sqrt(float(covs[1][0][0])))*weights[1] # 2nd gaussian
#y_axis2 = norm.pdf(x_axis, float(mean[2][0]), np.sqrt(float(covs[2][0][0])))*weights[2] # 1st gaussian
#y_axis3 = norm.pdf(x_axis, float(mean[3][0]), np.sqrt(float(covs[3][0][0])))*weights[3] # 2nd gaussian

ax = fig.add_subplot(1,2,2)
# Plot 2
plt.hist(X, density=True, color='black', bins=np.arange(0,300, 3))
plt.plot(x_axis, y_axis0, lw=6, c='C0',label="Peak: {:.2f}".format(float(mean[0][0])))
plt.plot(x_axis, y_axis1, lw=6, c='C1',label="Peak: {:.2f}".format(float(mean[1][0])))
#plt.plot(x_axis, y_axis2, lw=6, c='C2')
#plt.plot(x_axis, y_axis3, lw=6, c='C3')
plt.vlines(160,0,0.027, lw=4, ls='dashed', color='r', label="160")
plt.vlines(137,0,0.027, lw=4, ls='dashed', color='b', label="137")
#plt.vlines(119,0,0.027, lw=4, ls='dashed', color='r', label="119")
plt.plot(x_axis, y_axis0+y_axis1, lw=6, c='C2', ls='dashed') #, label="Peak: {:.2f}".format(x_axis[np.argmax(y_axis0+y_axis1)]))
plt.vlines(x_axis[np.argmax(y_axis0+y_axis1)],0,0.026, lw=4, ls='dashed', color='y', label="Peak: {:.2f}".format(x_axis[np.argmax(y_axis0+y_axis1)]))
#plt.plot(x_axis, y_axis0+y_axis1+y_axis2+y_axis3, lw=6, c='C2', ls='dashed')
#plt.xlim(50, 200)
#plt.ylim(0.0, 2.0)
plt.xlabel(r"X", fontsize=30)
plt.ylabel(r"Density", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=30)
plt.subplots_adjust(wspace=0.3)
plt.show()
plt.close('all')



