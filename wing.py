import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage
import sys
import os
import networkx as nx

from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import matplotlib.tri as mtri
import cmath
import matplotlib.colors as colors
from skimage import measure
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

from numpy import ones,vstack
from numpy.linalg import lstsq

sys.path.append("/Users/danyunhe/sknw")
import sknw

class wing:
    def __init__(self,filename):
        self.img=mpimg.imread(filename)
        # gray=np.load("./population_60+FMNH_4602368_hw_outline.npy")
        self.gray=np.load(filename)
        
        #vein:1 \\\\otherwise: 0
        binary=self.gray.copy()
        nnx=len(binary[0,:])
        ny=len(binary[:,0])
        for jj in range(0,ny):
            for ii in range(0,nnx):
                if binary[jj,ii]==0.5:
                    binary[jj,ii]=0
                    
        thresh=binary.astype('uint8')

        # get distance transform
        distance = thresh.copy()
        distance = cv2.distanceTransform(distance, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)
        
        # get skeleton (medial axis)
        binary2 = thresh.copy()
        skeleton = skimage.morphology.skeletonize(binary2).astype(np.float32)
        
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
        # Get all the nodes from the graph 
        self.all_nodes=G_subgraphs.nodes()
        self.N_nodes=len(self.all_nodes)
        #draw_vein_network(G_subgraphs)

    def unit_vector(self,vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self,v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    #given (ptx,pty)
    #find edge line segment close to pt,
    #and align with direction given by [dirlx,dirly][dirrx,dirry], l-left, r-right
    def find_lineup_edge(self,ptx,pty,dirlx,dirly,dirrx,dirry):
        mid_pt=np.array([ptx,pty]).astype(np.float32)
        #Find 10 closest nodes
        dis_list=np.zeros(self.N_nodes)
        for i in range(self.N_nodes):
            node_x=self.all_nodes[i]['o'][1]
            node_y=self.all_nodes[i]['o'][0]
            mid_pt_node_dis=np.sqrt((mid_pt[0]-node_x)**2+(mid_pt[1]-node_y)**2)
            dis_list[i]=mid_pt_node_dis
        asc_dis=np.argsort(dis_list)  
        #The first 20 of asc_dis gives the index of 20 closest nodes


        #Collect all edges incident to these 20 nodes
        edge_list=[]
        for i in range(20):
            node_id=asc_dis[i]
            edge_list=edge_list+list(G_subgraphs.edges(node_id))
        N_edges=len(edge_list)



        #First, sort edges based on distance to the given node
        #Then, loop through the edges:
            #If the edge direction is within a threshold, continue:
                #if the edge length is larger than a thresdhold:
                    #Choose this edge, Terminate
        edge_node_dis=np.zeros(N_edges)
        edge_length=np.zeros(N_edges)
        for i in range(N_edges):
            edge_node_id0=edge_list[i][0]
            edge_node_id1=edge_list[i][1]
            edge_x0=self.all_nodes[edge_node_id0]['o'][1].astype(np.float32)
            edge_y0=self.all_nodes[edge_node_id0]['o'][0].astype(np.float32)
            edge_x1=self.all_nodes[edge_node_id1]['o'][1].astype(np.float32)
            edge_y1=self.all_nodes[edge_node_id1]['o'][0].astype(np.float32)
            
            v=np.array([edge_x0,edge_y0]).astype(np.float32)
            w=np.array([edge_x1,edge_y1]).astype(np.float32)
            p=mid_pt
            
            l2=np.linalg.norm(w-v)
            edge_length[i]=l2
            l2=l2**2
            
            t= max(0, min(1, np.dot(p - v, w - v) / l2))
            projection = v + t * (w - v);
                
            lll=np.linalg.norm(p-projection)
            edge_node_dis[i]=lll
            
        length_thres=np.mean(edge_length)*0.5
        edge_node_dis_asc_ind=np.argsort(edge_node_dis) 

        vein_edge_id=-1
        end_pt_l=np.array([dirlx,dirly])
        end_pt_r=np.array([dirrx,dirry])
        #Loop through edges based on distance to node
        for i in edge_node_dis_asc_ind:
            #If edge length larger than threshold
            if edge_length[i]>length_thres:
                #If edge direction close to given nodes direction
                edge_node_id0=edge_list[i][0]
                edge_node_id1=edge_list[i][1]
                edge_x0=self.all_nodes[edge_node_id0]['o'][1].astype(np.float32)
                edge_y0=self.all_nodes[edge_node_id0]['o'][0].astype(np.float32)
                edge_x1=self.all_nodes[edge_node_id1]['o'][1].astype(np.float32)
                edge_y1=self.all_nodes[edge_node_id1]['o'][0].astype(np.float32)
                
                v1=np.array([edge_x0-edge_x1,edge_y0-edge_y1])
                v1=v1.astype(np.float32)
                if edge_x0>edge_x1:
                    v1=np.array([edge_x1-edge_x0,edge_y1-edge_y0]).astype(np.float32)
                v2=end_pt_l-end_pt_r
                angle=self.angle_between(v1, v2)
                if angle<np.pi/10:
                    vein_edge_id=i
                    break
        edge_node_id0=edge_list[vein_edge_id][0]
        edge_node_id1=edge_list[vein_edge_id][1]
        edge_x0=self.all_nodes[edge_node_id0]['o'][1].astype(np.float32)
        edge_y0=self.all_nodes[edge_node_id0]['o'][0].astype(np.float32)
        edge_x1=self.all_nodes[edge_node_id1]['o'][1].astype(np.float32)
        edge_y1=self.all_nodes[edge_node_id1]['o'][0].astype(np.float32)
        
        xl=edge_x0
        yl=edge_y0
        xr=edge_x1
        yr=edge_y1
        if edge_x1<edge_x0:
            xl=edge_x1
            yl=edge_y1
            xr=edge_x0
            yr=edge_y0
        return xl,yl, xr,yr



    def find_interp(self,lmk1,lmk2,lmk3,is_first_vein=False):
        #Collect the three points on/near vein
        mid_pt=np.zeros(2)
        mid_pt[0]=lmk2[0]
        mid_pt[1]=lmk2[1]
        mid_pt.astype(np.float32)

        end_pt_l=np.zeros(2)
        end_pt_l[0]=lmk1[0]
        end_pt_l[1]=lmk1[1]
        end_pt_l.astype(np.float32)

        end_pt_r=np.zeros(2)
        end_pt_r[0]=lmk3[0]
        end_pt_r[1]=lmk3[1]
        end_pt_r.astype(np.float32)  
        
        #Find the line y=mx+c through the vein found
        #find closest edge from graph to midpoint-ML
        #direction given by leftmost and rightmost points-ML
        edge_x0,edge_y0, edge_x1,edge_y1=self.find_lineup_edge(mid_pt[0],mid_pt[1],end_pt_l[0],end_pt_l[1],end_pt_r[0],end_pt_r[1])
        points = [(edge_x0,edge_y0),(edge_x1,edge_y1)]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]  
        
        #Find the projection point on the line near body
        p1=np.array([edge_x0,edge_y0])
        p2=np.array([edge_x1,edge_y1])
        p3=end_pt_r
        l2 = np.sum((p1-p2)**2)
        t = np.sum((p3 - p1) * (p2 - p1)) / l2
        projection_end_r = p1 + t * (p2 - p1)
        
        #Collect the vein line segments on the right hand side
        vein_line_seg_list=[]
        # vein_line_seg_list.append(projection_end_r)
        vein_line_seg_list.append(p2)
        vein_line_seg_list.append(p1)
        
        ## good for vein 2-7

        continue_search=True
        len_v=np.sqrt(abs(edge_x0-edge_x1)**2+abs(edge_y0-edge_y1))
        m=-1
        c=-1
        v0=np.array([end_pt_l[0]-mid_pt[0],end_pt_l[1]-mid_pt[1]])
        while continue_search:
            dir_vector=np.array([vein_line_seg_list[-1][0]-vein_line_seg_list[-2][0],vein_line_seg_list[-1][1]-vein_line_seg_list[-2][1]])
            dir_vector=dir_vector/np.linalg.norm(dir_vector)
            
            continue_calculate_newpt=True
            CTTT=1
            while continue_calculate_newpt:
                vein_pt_new=np.array([edge_x0,edge_y0]) + dir_vector*len_v*CTTT
                CTTT+=1
                gi=int(vein_pt_new[0])
                gj=int(vein_pt_new[1])
                
                if self.gray[gj,gi]==0.5: # if reach the background
                    continue_calculate_newpt=False
                    continue_search=False
                    
                    #Find the intersection point on the wing edge with the last line segment
                    continue_find_edge_point=True
                    xk=edge_x0
                    while continue_find_edge_point:
                        xk=xk-1
                        yk=m*xk+c
                        gi=int(xk)
                        gj=int(yk)
                        if self.gray[gj,gi]==0.5:
                            continue_find_edge_point=False
                            p1=np.array([xk,yk])
                            vein_line_seg_list.append(p1)
                    
                else: # if not reach the background, find the interior nodes
                    tmp_edge_x0,tmp_edge_y0, tmp_edge_x1,tmp_edge_y1=self.find_lineup_edge(vein_pt_new[0],vein_pt_new[1],vein_line_seg_list[-1][0],vein_line_seg_list[-1][1], vein_line_seg_list[-2][0],vein_line_seg_list[-2][1])
                    if(is_first_vein):
                        tmp_edge_x0,tmp_edge_y0, tmp_edge_x1,tmp_edge_y1=self.find_lineup_edge(vein_pt_new[0],vein_pt_new[1],end_pt_l[0],end_pt_l[1],vein_line_seg_list[-1][0],vein_line_seg_list[-1][1])
                    p1=np.array([tmp_edge_x0,tmp_edge_y0])
                    
                    #check if find the same edge 
                    if (p1[0]==vein_line_seg_list[-1][0] and p1[1]==vein_line_seg_list[-1][1]):
                        continue_calculate_newpt=True
                    else:
                        v1=np.array([tmp_edge_x0-tmp_edge_x1, tmp_edge_y0-tmp_edge_y1])
                        v2=np.array([vein_line_seg_list[-1][0]-vein_line_seg_list[-2][0], vein_line_seg_list[-1][1]-vein_line_seg_list[-2][1]])
                        angle_in_between=self.angle_between(v1, v2)
                        angle_in_between2=self.angle_between(v0,v1)
                        if(is_first_vein):
                            angle_criteria=angle_in_between>np.pi/10 or angle_in_between2>np.pi/10
                        else:
                            angle_criteria=angle_in_between>np.pi/15
                        if(angle_criteria): #check if the new edge found direction is too different
                            continue_calculate_newpt=True
                        else:
                            continue_calculate_newpt=False
                        
                            edge_x0=tmp_edge_x0
                            edge_y0=tmp_edge_y0 
                            edge_x1=tmp_edge_x1
                            edge_y1=tmp_edge_y1
                            p1=np.array([edge_x0,edge_y0])
                            vein_line_seg_list.append(p1)
                            
                            points = [(vein_line_seg_list[-1][0],vein_line_seg_list[-1][1]),(vein_line_seg_list[-2][0],vein_line_seg_list[-2][1])]
                            x_coords, y_coords = zip(*points)
                            A = vstack([x_coords,ones(len(x_coords))]).T
                            m, c = lstsq(A, y_coords)[0]
                            
        vein_line_seg_list=np.array(vein_line_seg_list)
        
        return vein_line_seg_list



if __name__=="__main__":

    img_idx=10
    dir="./images/"
    # img=cv2.imread(dir+'0.png')
    img0 = mpimg.imread(dir+'%d.png'%img_idx)

    ny0=len(img0[:, 0])  # y
    nx0=len(img0[0, :])  # x
    
    lm_idx=1
    lmk1=[img_x[lm_idx-1],img_y[lm_idx-1]]
    lmk2=[img_x[lm_idx+11+8],img_y[lm_idx+11+8]]
    lmk3=[img_x[lm_idx+11+1],img_y[lm_idx+11+1]]
    
    
    a=wing("filename")
    pt=a.find_interp(....)
    
    a.map()
    
    