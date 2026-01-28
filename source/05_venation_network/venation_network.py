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
        self.img0=mpimg.imread(os.path.join(self.svd_dir,'population_{}'.format(self.population),'population_{}+FMNH_{}_hw_1.png'.format(self.population, self.species)))    #test image
        self.ny0=len(self.img0[:, 0])  # y
        self.nx0=len(self.img0[0, :])  # x
        
        # ====================== Read in image that categorize vein 1, cell 0, background 0.5.
        self.vein_cell_bg=np.load(os.path.join(self.input_dir,'population_{}'.format(self.population),'population_{}+FMNH_{}_hw_outline.npy'.format(self.population, self.species)))    #test image
        
        # ====================== Read in cellpose segmentation info
        seg_path = os.path.join(
                                self.input_dir,
                                'population_{}'.format(self.population),
                                'population_{}+FMNH_{}_hw_seg.npy'.format(
                                    self.population, self.species
                                )
                            )
        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"No cellpose segmentation file found: {seg_path}")

        self.wing_cellpose = np.load(seg_path, allow_pickle=True).item()
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


def find_all_wings(input_dir):
    """
    Find all wing base names from Step 4 segmentation output.

    Args:
        input_dir: Directory containing segmentation output (e.g., ../../result/04_segmentation/)

    Returns:
        List of base names (e.g., ["population_60+FMNH_4602398", ...])
    """
    pattern = os.path.join(input_dir, "population_*", "*_hw_outline.npy")
    outline_files = glob.glob(pattern)

    if not outline_files:
        print(f"Warning: No segmentation outputs found matching pattern: {pattern}")
        return []

    base_names = []
    for f in outline_files:
        basename = os.path.basename(f)
        # Remove "_hw_outline.npy" to get base name like "population_60+FMNH_4602398"
        base_name = basename.replace("_hw_outline.npy", "")
        base_names.append(base_name)

    return sorted(base_names)


def process_single_wing(base_name, svd_dir, input_dir, output_dir,
                        save_venation_network, save_plot, save_data):
    """
    Run venation network analysis on a single wing.

    Args:
        base_name: Wing identifier (e.g., "population_60+FMNH_4602398")
        svd_dir: SVD output directory
        input_dir: Segmentation output directory
        output_dir: Venation network output directory
        save_venation_network: Whether to save the network graph
        save_plot: Whether to save plots
        save_data: Whether to save stats as .txt

    Returns:
        True if successful, False otherwise
    """
    # Parse population and species from base name
    # e.g., "population_60+FMNH_4602398" -> population="60", species="4602398"
    parts = base_name.split("+")
    population = parts[0].replace("population_", "")
    species = parts[1].replace("FMNH_", "")

    # Create and analyze the wing object
    wing = wing_venation_network(
        population=population,
        species=species,
        svd_dir=svd_dir,
        input_dir=input_dir,
        save_dir=output_dir
    )

    # Compute basic measurements
    wing.calculate_wing_area()
    wing.calculate_domain_stats()
    wing.calculate_venetion_network()
    wing.calculate_modularity_communities()

    # Save venation network
    if save_venation_network:
        wing.save_venation_network()

    # Save plots
    if save_plot:
        wing.plot_cells_fractional_area()
        wing.plot_cells_circularity()
        wing.plot_vein_network()
        wing.plot_vein_thickness()
        wing.plot_modularity_communities()

    # Save stats as .txt
    if save_data:
        wing.save_stats_as_txt()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build venation networks from segmented wing images"
    )
    parser.add_argument("--svd_dir", default="../../result/03_svd/",
                        help="SVD output directory (default: ../../result/03_svd/)")
    parser.add_argument("--input_dir", default="../../result/04_segmentation/",
                        help="Segmentation output directory (default: ../../result/04_segmentation/)")
    parser.add_argument("--output_dir", default="../../result/05_venation_network/",
                        help="Venation network output directory (default: ../../result/05_venation_network/)")
    parser.add_argument("--wing_name", type=str,
                        help="Process only this wing (e.g., population_34+FMNH_4669630)")
    parser.add_argument("--population_id", type=int,
                        help="Process only this population (e.g., 34)")

    # Save flags
    parser.add_argument("--save_venation_network", type=str2bool, default=True,
                        help="Save venation network graph (default: True)")
    parser.add_argument("--save_plot", type=str2bool, default=True,
                        help="Save plots (default: True)")
    parser.add_argument("--save_data", type=str2bool, default=True,
                        help="Save stats to .txt (default: True)")

    args = parser.parse_args()

    print(f"SVD directory: {os.path.abspath(args.svd_dir)}")
    print(f"Input directory: {os.path.abspath(args.input_dir)}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")

    # Discover wings from Step 4 output
    all_wings = find_all_wings(args.input_dir)

    if not all_wings:
        print(f"Error: No wings found in {args.input_dir}")
        print("Please run Step 4 (segmentation.py) first.")
        sys.exit(1)

    # Filter by specific wing or population if requested
    if args.wing_name:
        if args.wing_name in all_wings:
            wings_to_process = [args.wing_name]
            print(f"\nProcessing specific wing: {args.wing_name}")
        else:
            print(f"Error: Wing '{args.wing_name}' not found in segmentation output")
            print(f"Available wings: {all_wings[:5]}...")
            sys.exit(1)
    elif args.population_id is not None:
        pop_prefix = f"population_{args.population_id}+"
        wings_to_process = [w for w in all_wings if w.startswith(pop_prefix)]
        if not wings_to_process:
            print(f"Error: No wings found for population {args.population_id}")
            sys.exit(1)
        print(f"\nProcessing population {args.population_id}: {len(wings_to_process)} wing(s)")
    else:
        wings_to_process = all_wings
        print(f"\nFound {len(wings_to_process)} wing(s) to process")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each wing
    successful = 0
    failed = 0
    failed_wings = []

    for i, wing_name in enumerate(wings_to_process, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(wings_to_process)}] Processing: {wing_name}")
        print(f"{'='*60}")

        try:
            if process_single_wing(wing_name, args.svd_dir, args.input_dir,
                                   args.output_dir, args.save_venation_network,
                                   args.save_plot, args.save_data):
                print(f"  Success: {wing_name}")
                successful += 1
            else:
                print(f"  Failed: {wing_name}")
                failed += 1
                failed_wings.append(wing_name)
        except Exception as e:
            print(f"  Failed: {wing_name} - {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            failed_wings.append(wing_name)

    # Summary
    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if failed_wings:
        print(f"Failed wings: {', '.join(failed_wings)}")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()


