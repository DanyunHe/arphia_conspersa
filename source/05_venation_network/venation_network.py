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

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environment
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.collections import PatchCollection
import matplotlib.patches #import Polygon
from matplotlib.collections import LineCollection



class wing_venation_network:
    
    #Set up paths to input and output data
    def __init__(self, population, species, svd_dir, input_dir, save_dir):

        # Specify SVD directory for PNG images
        self.svd_dir=svd_dir
        # Specify input directory for segmentation files
        self.input_dir=input_dir
        # Specify directory to save results
        self.save_dir=os.path.join(save_dir, "population_{}".format(population))
        #Specify this image population and species
        self.population=population
        self.species=species
        
        # Make output directory if it doesn't exist
        self.save_dir_cell = os.path.join(self.save_dir, "cell")
        self.save_dir_vein = os.path.join(self.save_dir, "vein")
        os.makedirs(self.save_dir_cell, exist_ok=True)
        os.makedirs(self.save_dir_vein, exist_ok=True)

        
        
         
        # ====================== Read in wing image
        self.img0=mpimg.imread(os.path.join(self.svd_dir,'population_{}+FMNH_{}_hw_1.png'.format(self.population, self.species)))    #test image
        self.ny0=len(self.img0[:, 0])  # y
        self.nx0=len(self.img0[0, :])  # x
        
        # ====================== Read in image that categorize vein 1, cell 0, background 0.5. 
        self.vein_cell_bg=np.load(os.path.join(self.input_dir,'population_{}+FMNH_{}_hw_outline.npy'.format(self.population, self.species)))    #test image
        
        # ====================== Read in cellpose segmentation info
        pattern = os.path.join(
                                self.input_dir,
                                'population_{}+FMNH_{}_seg.npy'.format(
                                    self.population, self.species
                                )
                            )
        # Search for matching file
        matches = glob.glob(pattern)
        
        if len(matches) == 0:
            raise FileNotFoundError(f"No cellpose segmentation file found for population {self.population}, species {self.species}")

        
        self.wing_cellpose = np.load(matches[0], allow_pickle=True).item()
        self.cell_contours=utils.outlines_list(self.wing_cellpose['masks'])
        
        
        #Some statistical variables
        self.num_domains=len(self.cell_contours)
        self.cells_polygon=[]
        
        self.wing_area=0
        
        self.cells_area=[]
        self.cells_fractional_area=None
        self.cells_perimeter=[]
        self.cells_circularity=None
        
        self.average_vein_thickness = None
        self.venation_network = None
        
        

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
        plt.savefig(os.path.join(self.save_dir_cell,"population_{}+FMNH_{}_hw_cell_fractional_areas.png".format(self.population,self.species)))
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
        plt.savefig(os.path.join(self.save_dir_cell,"population_{}+FMNH_{}_hw_cell_circularity.png".format(self.population,self.species)))
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
        self.average_vein_thickness = np.median(thickness[skeleton!=0])
    
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
        
        self.venation_network=nx.compose_all(G_sub)
        
        #Associate thickness and length attributes to each edge
        for (s,e) in self.venation_network.edges():
            ps = self.venation_network[s][e]['pts']
            avg_edge_thickness=0.0
            ei_npt=len(ps)
            for ei in range(0,ei_npt):
                eix=ps[ei][1]
                eiy=ps[ei][0]
                avg_edge_thickness=avg_edge_thickness+thickness[eiy,eix]
            avg_edge_thickness=avg_edge_thickness/ei_npt
            
            self.venation_network[s][e]['weight'] = avg_edge_thickness
            self.venation_network[s][e]['length'] = ei_npt
            
        # find boundary edges, nodes
        self._find_graph_boundary()
        
    def calculate_modularity_communities(self):
        
        self.modularity_communities = nx.community.greedy_modularity_communities(self.venation_network)
        self.num_communities=len(self.modularity_communities)
        
            
    def save_venation_network(self):
        # save graph object to file
        save_path = os.path.join(self.save_dir_vein,f"population_{self.population}+FMNH_{self.species}_hw_vein_graph.pickle")
        with open(save_path, 'wb') as f:
            pickle.dump(self.venation_network, f, pickle.HIGHEST_PROTOCOL)
   

    def save_stats_as_txt(self):
        """
        Save computed cell statistics as .txt files in the cell subfolder.
        """
        # ----- Save cell stats -----
        if self.cells_area is not None:
            np.savetxt(os.path.join(self.save_dir_cell, f"population_{self.population}+FMNH_{self.species}_hw_cells_area.txt"), self.cells_area)
        if self.cells_perimeter is not None:
            np.savetxt(os.path.join(self.save_dir_cell, f"population_{self.population}+FMNH_{self.species}_hw_cells_perimeter.txt"), self.cells_perimeter)
        if self.cells_circularity is not None:
            np.savetxt(os.path.join(self.save_dir_cell, f"population_{self.population}+FMNH_{self.species}_hw_cells_circularity.txt"), self.cells_circularity)
        if self.cells_fractional_area is not None:
            np.savetxt(os.path.join(self.save_dir_cell, f"population_{self.population}+FMNH_{self.species}_hw_cells_fractional_area.txt"), self.cells_fractional_area)

        # ----- Save wing-level summary -----
        with open(os.path.join(self.save_dir, f"population_{self.population}+FMNH_{self.species}_hw_summary.txt"), 'w') as f:
            f.write(f"Population: {self.population}\n")
            f.write(f"Species: {self.species}\n")
            f.write(f"Wing area: {self.wing_area}\n")
            f.write(f"Number of domains: {self.num_domains}\n")
            if self.average_vein_thickness is not None:
                f.write(f"Average vein thickness: {self.average_vein_thickness:.3f}\n")
            if hasattr(self, "modularity_communities"):
                f.write(f"Number of modularity communities: {len(self.modularity_communities)}\n")
            
            
    def plot_modularity_communities(self):
        # Prepare color map
        colors = cm.gist_rainbow(np.linspace(0, 1, self.num_communities))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(32, 23))
        
        # Draw edges
        for s, e in self.venation_network.edges():
            ps = self.venation_network[s][e]['pts']
            ax.plot(ps[:, 1], ps[:, 0], color='black', linewidth=1)
        
        # Draw nodes by community
        for idx, community in enumerate(self.modularity_communities):
            for node in community:
                coord = np.array([self.venation_network.nodes[node]['o']])
                ax.scatter(coord[:, 1], coord[:, 0], s=500, color=colors[idx], label=f"Community {idx}")
        
        # Finalize plot
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title("Maximum modularity communities", fontsize=50)
        
        # Save and show
        save_path = os.path.join(self.save_dir_vein, f"population_{self.population}+FMNH_{self.species}_hw_max_mod_communities.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
                
           
    def plot_vein_network(self):
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(38,23))
    
        # Invert y-axis to match image coordinate system
        ax.invert_yaxis()
    
        # Draw edges (veins) using stored points
        for s, e in self.venation_network.edges():
            ps = self.venation_network[s][e]['pts']
            ax.plot(ps[:, 1], ps[:, 0], color='green', linewidth=1)
    
        # Draw nodes (junctions)
        nodes = self.venation_network.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        ax.scatter(ps[:, 1], ps[:, 0], s=70, c="red", zorder=5)
    
        # Clean up axis and save
        ax.set_title('Venation network')
        ax.axis('off')
        save_path = os.path.join(self.save_dir_vein, f"population_{self.population}+FMNH_{self.species}_hw_venation_network.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        
    def plot_vein_thickness(self):
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(38,23))

        # Prepare line segments and their thickness (color array)
        segx, segy, clist = [], [], []
        for s, e in self.venation_network.edges():
            ps = self.venation_network[s][e]['pts']
            segx.append(ps[:, 1].tolist())  # x = column
            segy.append(ps[:, 0].tolist())  # y = row
            clist.append(self.venation_network[s][e]['weight'])
    
        segments = [np.column_stack([x, y]) for x, y in zip(segx, segy)]
        lc = LineCollection(segments, cmap="rainbow", linewidth=8)
        lc.set_array(np.array(clist))
        lc.set_clim([10, 20])
    
        # Add to plot
        im = ax.add_collection(lc)
        ax.autoscale_view()
        ax.axis('off')
    
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=12)
    
        ax.invert_yaxis()
        ax.set_title("Vein thickness")
    
        # Save figure
        save_path = os.path.join(self.save_dir_vein,f"population_{self.population}+FMNH_{self.species}_hw_vein_thickness.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
            
        
    def _find_graph_boundary(self):
        nodes=self.venation_network.nodes()
        nx.set_node_attributes(self.venation_network, 0, "boundary")
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
            for (s,e) in self.venation_network.edges(ni): 
                edge_coords=self.venation_network[s][e]['pts']  
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
            for (s,e) in self.venation_network.edges(ni): 
                edge_coords=self.venation_network[s][e]['pts']  
                self.venation_network[s][e]['boundary']=0
                edge_mid_loc=int(len(edge_coords)/2)
                ptmy,ptmx=edge_coords[edge_mid_loc]
                
                ptiy, ptix =edge_coords[0]
                ptey, ptex =edge_coords[-1]
                perp_vec=np.asarray([ptiy-ptey,ptex-ptix])
                if np.abs(ptiy-ptey)<1:
                    perp_vec=np.asarray([0,1])
                
                perp_vec=perp_vec/np.linalg.norm(perp_vec)
                perx,pery=perp_vec
                
                dis=min(max(self.venation_network[s][e]['length']/2,3),10)
                
                testy,testx=int(ptmy+dis*pery),int(ptmx+dis*perx)
                if background_temp[testy,testx]==0.5:
                    self.venation_network[s][e]['boundary']=1
                    nodes[ni]['boundary']=1
                    
                else:
                    testy,testx=int(ptmy-dis*pery),int(ptmx-dis*perx)
                    if background_temp[testy,testx]==0.5:
                        self.venation_network[s][e]['boundary']=1
                        nodes[ni]['boundary']=1
        #Loop through edges again, make sure no bdry edge left un-accounted
        for ni in nodes:
            #Incident edges to a node
            for (s,e) in self.venation_network.edges(ni): 
                if nodes[s]['boundary']==1 and nodes[e]['boundary']==1:
                    self.venation_network[s][e]['boundary']=1
        
        
        
        
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "True", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "False", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

        
        
if __name__ == '__main__':
    """
    Example Command-Line Usage:
        Data and plots saving default are true:
            
            python3 venation_network.py \
            --input_dir 04_segmentation_output \
            --output_dir 05_venation_network_output \
            --population 151 --species 4601939 \
        
        Not saving: 
            python3 venation_network.py \
            --input_dir 04_segmentation_output \
            --output_dir 05_venation_network_output \
            --population 151 --species 4601939 \
            --save_venation_network False \ 
            --save_plot False \
            --save_data False
    """
    
    # Argument parser setup
    parser = argparse.ArgumentParser(
                description="""\
                            Example Command-Line Usage:
                            
                                Data and plots saving default are True:
                                    python3 venation_network.py \\
                                        --input_dir 04_segmentation_output \\
                                        --output_dir 05_venation_network_output \\
                                        --population 151 --species 4601939
                            
                                Not saving:
                                    python3 venation_network.py \\
                                        --input_dir 04_segmentation_output \\
                                        --output_dir 05_venation_network_output \\
                                        --population 151 --species 4601939 \\
                                        --save_venation_network False \\
                                        --save_plot False \\
                                        --save_data False
                            """,
                formatter_class=argparse.RawDescriptionHelpFormatter
                )

    
    
    # Input/output
    parser.add_argument("--svd_dir", type=str, required=True, help="Path to the SVD directory (for PNG images)")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the segmentation directory (for outline and cellpose files)")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    
    # Metadata for identifying the wing
    parser.add_argument("--population", type=str, required=True, help="Population ID")
    parser.add_argument("--species", type=str, required=True, help="Species ID")

    # Save flags
    parser.add_argument("--save_venation_network", type=str2bool, default=True, help="Default True; Save venation network or not")
    parser.add_argument("--save_plot", type=str2bool, default=True, help="Default True; Save plots or not")
    parser.add_argument("--save_data", type=str2bool, default=True, help="Default True; Save stats to .txt or not")
    
    args = parser.parse_args()
    
    
    # Create and analyze the wing object
    wing = wing_venation_network(
        population=args.population,
        species=args.species,
        svd_dir=args.svd_dir,
        input_dir=args.input_dir,
        save_dir=args.output_dir
    )
    print("created wing object")

    # Compute basic measurements
    wing.calculate_wing_area()
    wing.calculate_domain_stats()
    wing.calculate_venetion_network()
    wing.calculate_modularity_communities()
    print("finished calculation")

    # Save venation network
    if args.save_venation_network:
        wing.save_venation_network()
        print("saved venation network")

    # Save plots
    if args.save_plot:
        wing.plot_cells_fractional_area()
        wing.plot_cells_circularity()
        wing.plot_vein_network()
        wing.plot_vein_thickness()
        wing.plot_modularity_communities()
        print("saved plots")

    # Save stats as .txt
    if args.save_data:
        wing.save_stats_as_txt()
        print("saved data")
        
    
    
    
    
    
    
    
    
    
        
        
    
    
    
    
        
    
    
    
    
    
    
    
    




