


#utils

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 0/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)








if __name__ == '__main__':
    
    #SAM MODEL
        
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamPredictor
    
    sam_checkpoint = file_path_base+"fine_tuned_sam_im1b.pth"
    model_type = "vit_h"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    predictor = SamPredictor(sam)


    
    #POINTS EXCEL 
    

    
    population_id=44
    file_path_base="/content/drive/My Drive/wing/"
    file_path_main=file_path_base+"wing_new_populations/"
    file_path_img =file_path_main +"population_{}_original/".format(population_id)
    file_path_save =file_path_main+"images_masks_population_{}/".format(population_id)
    #population_excel = pd.read_excel(file_path_base+"extract_wings_populations.xlsx", sheet_name="population_{}".format(population_id))
    population_excel = pd.read_csv(file_path_main+"whole_label_{}.csv".format(population_id),header=1)
    N_wing=len(population_excel["bodyparts"][1:])
    
    
    if not os.path.exists(file_path_save):
       os.makedirs(file_path_save)
    if not os.path.exists(file_path_save+"problematic_full/"):
       os.makedirs(file_path_save+"problematic_full/")
    #if not os.path.exists(file_path_save+"sticky_full/"):
    #   os.makedirs(file_path_save+"sticky_full/")
    #if not os.path.exists(file_path_save+"sticky_cropped/"):
    #   os.makedirs(file_path_save+"sticky_cropped/")
    if not os.path.exists(file_path_save+"perfect_full/"):
       os.makedirs(file_path_save+"perfect_full/")
    if not os.path.exists(file_path_save+"perfect_cropped/"):
       os.makedirs(file_path_save+"perfect_cropped/")
    #if not os.path.exists(file_path_save+"missing_full/"):
    #   os.makedirs(file_path_save+"missing_full/")
    #if not os.path.exists(file_path_save+"missing_cropped/"):
    #   os.makedirs(file_path_save+"missing_cropped/")



    ### EXTRACTION
    for wi in range(32,N_wing+1):
    
        w_name_base=population_excel["bodyparts"][wi][:-6]
        w_name_0=w_name_base+"_0"
        w_name_1=w_name_base+"_1"
        
        print("{}/{}: {}\n".format(wi, N_wing,w_name_0+".dng"))
        
        with rawpy.imread(file_path_img+w_name_0+".dng") as raw:
            im_0 = raw.postprocess()
        with rawpy.imread(file_path_img+w_name_1+".dng") as raw:
            im_1 = raw.postprocess()
        
        bg_x=float(population_excel["background"][wi])
        bg_y=float(population_excel["background.1"][wi])
        
        hw_x0=float(population_excel["hind1"][wi])
        hw_y0=float(population_excel["hind1.1"][wi])
        fw_x0=float(population_excel["fore1"][wi])
        fw_y0=float(population_excel["fore1.1"][wi])
        
        hw_x1=float(population_excel["hind2"][wi])
        hw_y1=float(population_excel["hind2.1"][wi])
        fw_x1=float(population_excel["fore2"][wi])
        fw_y1=float(population_excel["fore2.1"][wi])
        
        
        body_x0=float(population_excel["bodypart1"][wi])
        body_y0=float(population_excel["bodypart1.1"][wi])
        body_x1=float(population_excel["bodypart2"][wi])
        body_y1=float(population_excel["bodypart2.1"][wi])
        body_x2=float(population_excel["bodypart3"][wi])
        body_y2=float(population_excel["bodypart3.1"][wi])
        
        extract_wing_im(predictor, im_0, 0, w_name_0, bg_x, bg_y, hw_x0,hw_y0,fw_x0,fw_y0,hw_x1,hw_y1,fw_x1,fw_y1,body_x0,body_y0,body_x1,body_y1,body_x2,body_y2)
        extract_wing_im(predictor, im_1, 1, w_name_1, bg_x, bg_y, hw_x0,hw_y0,fw_x0,fw_y0,hw_x1,hw_y1,fw_x1,fw_y1,body_x0,body_y0,body_x1,body_y1,body_x2,body_y2)
        
        
        




    ### CROPPING















