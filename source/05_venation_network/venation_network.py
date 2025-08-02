import glob
import pickle
import argparse
import os
import sys
dirname = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(dirname, '../sknw-master')))

import numpy as np
import cv2
import skimage.morphology
import sknw
import networkx as nx
from cellpose import utils
import shapely.geometry# import Polygon

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.collections import PatchCollection
import matplotlib.patches #import Polygon
from matplotlib.collections import LineCollection




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
        
        
        #Some statistical variables
        self.num_domains=len(self.cell_contours)
        self.wing_area=0
        self.cells_area=[]
        self.cells_fractional_area=None
        
        self.cells_polygon=[]
        self.cells_perimeter=[]
        self.cells_circularity=None
        
        

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
        self.cells_circularity=4*np.pi*self.cells_area/(self.cells_perimeter**2)
        self.cells_fractional_area=self.cells_area/self.wing_area
        
    
    def plot_cells_fractional_area(self):
        fig,ax = plt.subplots(1,figsize=(40, 30))  
        ax.imshow(self.img0)  #Plot against the wing image
    
        norm = matplotlib.colors.Normalize(vmin=0, vmax=0.006)
        cmap = plt.get_cmap('plasma')
        colors = cmap(norm(self.cells_fractional_area))
    
        patches = []
        for o in self.cell_contours:
            if len(o)>0:
                coords=[]
                for ii in range(0,len(o)):
                    coords.append(o[ii])
                coords.append(o[0])
                patches.append(matplotlib.patches.Polygon(coords,closed=True))
                
        collection = PatchCollection(patches)
        ax.add_collection(collection)
        collection.set_color(colors)
        
        ax.autoscale_view()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        c_map_ax = fig.add_axes([0.9, 0.1, 0.04, 0.8])
        cbar=matplotlib.colorbar.ColorbarBase(c_map_ax, orientation='vertical', 
                                cmap='plasma',
                                norm=mpl.colors.Normalize(0, 0.006))


        cbar.set_ticks(ticks=[0,0.001,0.002,0.003,0.004,0.005,0.006], \
                        labels=['0', '0.001','0.002','0.003','0.004', '0.005','0.006'])

        cbar.ax.tick_params(labelsize=40)
    
        ax.set_title("population_{}+FMNH_{}_hw\ncell fractional areas".format(self.population,self.species),fontsize=50)
        plt.savefig(self.save_dir+"/population_{}+FMNH_{}_hw_cell_fractional_areas.png".format(self.population,self.species))
        plt.close()
        
    
    def plot_cells_circularity(self):
        fig,ax = plt.subplots(1,figsize=(40, 30))   
        ax.imshow(self.img0)  #Plot against the wing image
        
        patches = []
        
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        cmap = plt.get_cmap('magma')
        colors = cmap(norm(self.cells_circularity))
        
        for o in self.cell_contours:
            if len(o)>0:
                coords=[]
                for ii in range(0,len(o)):
                    coords.append(o[ii])
                coords.append(o[0])
                patches.append(matplotlib.patches.Polygon(coords,closed=True))
        

        collection = PatchCollection(patches)
        ax.add_collection(collection)
        collection.set_color(colors)
        ax.autoscale_view()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        
        
        c_map_ax = fig.add_axes([0.9, 0.1, 0.04, 0.8])
        cbar=matplotlib.colorbar.ColorbarBase(c_map_ax, orientation='vertical', 
                                cmap='magma',
                                norm=mpl.colors.Normalize(0, 1))
        
        cbar.set_ticks(ticks=[0,0.25,0.5,0.75,1], \
                        labels=['0', '0.25','0.5','0.75','1'])
        cbar.ax.tick_params(labelsize=40)
        
        ax.set_title("population_{}+FMNH_{}_hw\ncell circularity\n".format(self.population,self.species),fontsize=50)
        plt.savefig(self.save_dir+"/population_{}+FMNH_{}_hw_cell_circularity.png".format(self.population,self.species))
        plt.close()
        
        
    def calculate_venetion_network(self):
        binary=self.vein_cell_bg.copy()
        
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
        average_thickness = np.median(thickness[skeleton!=0])
    
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

    




