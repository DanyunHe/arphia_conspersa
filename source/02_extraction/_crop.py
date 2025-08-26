










def crop_wing_func(wing_0_im,wing_1_im, file_save_crop,sorted_file_list_full_0, sorted_file_list_full_1):
    w_0=cv2.cvtColor(wing_0_im, cv2.COLOR_BGR2GRAY)
    w_1=cv2.cvtColor(wing_1_im, cv2.COLOR_BGR2GRAY)

    nx0=len(w_0[1,:])
    ny0=len(w_0[:,1])
    nx1=len(w_1[1,:])
    ny1=len(w_1[:,1])

    min_x_w=0
    continue_search=True
    ii=0
    while continue_search:
        all_white  = False if False in [x == 255 for x in w_0[:,ii]] else True
        if all_white==False:
            continue_search=False
            min_x_w=ii
        ii=ii+1

    max_x_w=nx0-1
    continue_search=True
    ii=nx0-1
    while continue_search:
        all_white  = False if False in [x == 255 for x in w_0[:,ii]] else True
        if all_white==False:
            continue_search=False
            max_x_w=ii
        ii=ii-1


    min_y_w=0
    continue_search=True
    jj=0
    while continue_search:
        all_white  = False if False in [x == 255 for x in w_0[jj,:]] else True
        if all_white==False:
            continue_search=False
            min_y_w=jj
        jj=jj+1

    max_y_w=ny0-1
    continue_search=True
    jj=ny0-1
    while continue_search:
        all_white  = False if False in [x == 255 for x in w_0[jj,:]] else True
        if all_white==False:
            continue_search=False
            max_y_w=jj
        jj=jj-1


    #crop wing, with a margin of +/- 50 pixels if possible
    if max_x_w>=min(nx0,nx1):
        max_x_w=min(nx0,nx1)
    if max_y_w>=min(ny0,ny1):
        max_y_w=min(ny0,ny1)

    if min_x_w-50>=0:
        min_x_w=min_x_w-50
    if min_y_w-50>=0:
        min_y_w=min_y_w-50
    if max_x_w+50<min(nx0,nx1):
        max_x_w=max_x_w+50
    if max_y_w+50<min(ny0,ny1):
        max_y_w=max_y_w+50

    #img[y:y+h, x:x+w]
    w_0_im_crop=wing_0_im[min_y_w:max_y_w, min_x_w:max_x_w]
    w_1_im_crop=wing_1_im[min_y_w:max_y_w, min_x_w:max_x_w]

    #save cropped wing
    cv2.imwrite(file_save_crop+os.path.splitext(sorted_file_list_full_0)[0][:-5]+"_crop.tif", w_0_im_crop)
    cv2.imwrite(file_save_crop+os.path.splitext(sorted_file_list_full_1)[0][:-5]+"_crop.tif", w_1_im_crop)


    #RECORD
    wing_category=os.path.splitext(sorted_file_list_full_0)[0][-7:-9]
    #record in excel: original dimensions, crop locations, after crop dimensions
    append_data = pd.DataFrame([{'filename':os.path.splitext(sorted_file_list_full_0)[0][:-5],
                                 'w0':nx0,
                                 'h0':ny0,
                                 'w1':nx1,
                                 'h1':ny1,

                                 'c_ax_{}'.format(wing_category):min_x_w,
                                 'c_bx_{}'.format(wing_category):max_x_w,
                                 'c_ay_{}'.format(wing_category):min_y_w,
                                 'c_by_{}'.format(wing_category):max_y_w,
                                 'cw_{}'.format(wing_category):w_0_im_crop.shape[1],
                                 'ch_{}'.format(wing_category):w_0_im_crop.shape[0],

                                 'folder':file_folder

                                 }])

    if os.path.isfile(file_path_save+"cropping_log_population_{}.xlsx".format(population_id)):
        wb = load_workbook(file_path_save+"cropping_log_population_{}.xlsx".format(population_id))
    else:
        wb = openpyxl.Workbook()
        wb.save(filename =file_path_save+"cropping_log_population_{}.xlsx".format(population_id))
        wb = load_workbook(filename =file_path_save+"cropping_log_population_{}.xlsx".format(population_id))
    if not "crop_{}".format(wing_category) in wb.sheetnames:
        wb.create_sheet("crop_{}".format(wing_category))
        ws = wb["crop_{}".format(wing_category)]
        ws.append(["filename", "w0", "h0", "w1", "h1", 'c_ax_{}'.format(wing_category), 'c_bx_{}'.format(wing_category), 'c_ay_{}'.format(wing_category),'c_by_{}'.format(wing_category),'cw_{}'.format(wing_category),'ch_{}'.format(wing_category),'folder'])

    ws = wb["crop_{}".format(wing_category)]

    for r in dataframe_to_rows(append_data, index=False, header=False):  #No index and don't append the column headers
        ws.append(r)
    wb.save(file_path_save+"cropping_log_population_{}.xlsx".format(population_id))
    
    
    
    
    
    
    
### BATCH CROPPING
pid_list=[145,151,161,173]
for population_id in pid_list:
  file_path_base="/content/drive/My Drive/wing/"
  file_path_main=file_path_base+"wing_new_populations/"
  file_path_img =file_path_main +"population_{}_original/".format(population_id)
  file_path_save =file_path_main+"images_masks_population_{}/".format(population_id)
  population_excel = pd.read_csv(file_path_main+"whole_label_{}.csv".format(population_id),header=1)
  N_wing=len(population_excel["bodyparts"][1:])

  if not os.path.exists(file_path_save+"perfect_cropped/"):
    os.makedirs(file_path_save+"perfect_cropped/")

  for file_i in range(1):
    file_category=["perfect"][file_i]
    # Read in full size segmented wing files
    file_folder_read="{}_full/".format(file_category)
    file_path_read=file_path_save+file_folder_read
    file_path_full = os.listdir(file_path_read)
    sorted_file_list_full=sorted(file_path_full)
    num_wings_crop=int(len(sorted_file_list_full)/4)
    if file_category=="missing":
      num_wings_crop=int(len(sorted_file_list_full)/2)

    #Save file locations
    file_folder="{}_cropped/".format(file_category)
    file_save_crop=file_path_save+file_folder
    for i in range(0,num_wings_crop):

      print(file_category+":{}/{}\n".format(i, num_wings_crop))

      if file_category=="missing":
        #READ DATA
        w_0_im=cv2.imread(file_path_read + sorted_file_list_full[2*i],-1)
        w_1_im=cv2.imread(file_path_read + sorted_file_list_full[2*i+1],-1)
        #CROP
        crop_wing_func(w_0_im,w_1_im, file_save_crop,sorted_file_list_full[2*i],sorted_file_list_full[2*i+1])


      if file_category=="perfect" or file_category=="select_perfect":

        #READ DATA
        fw_0_im=cv2.imread(file_path_read + sorted_file_list_full[4*i],-1)
        hw_0_im=cv2.imread(file_path_read + sorted_file_list_full[4*i+1],-1)
        fw_1_im=cv2.imread(file_path_read + sorted_file_list_full[4*i+2],-1)
        hw_1_im=cv2.imread(file_path_read + sorted_file_list_full[4*i+3],-1)

        #CLEAR DISCONNECTED COMPONENTS
        population_excel = pd.read_csv(file_path_main+"whole_label_{}.csv".format(population_id),header=1)
        wi=population_excel.loc[population_excel["bodyparts"]==sorted_file_list_full[4*i][:-13] + "1.dng"].index[0]
        fw_x0=int(float(population_excel["fore1"][wi]))
        fw_y0=int(float(population_excel["fore1.1"][wi]))
        hw_x1=int(float(population_excel["hind2"][wi]))
        hw_y1=int(float(population_excel["hind2.1"][wi]))


        bgmask = flood(hw_1_im[..., 1], (1,1), tolerance=0.1)
        bgmask = bgmask.astype(int)
        wingmask=flood(bgmask, (hw_y1,hw_x1), tolerance=0.1)
        hw_1_im[wingmask==False]=(255,255,255)

        bgmask = flood(hw_0_im[..., 1], (1,1), tolerance=0.1)
        bgmask = bgmask.astype(int)
        wingmask=flood(bgmask, (hw_y1,hw_x1), tolerance=0.1)
        hw_0_im[wingmask==False]=(255,255,255)

        bgmask = flood(fw_1_im[..., 1], (1,1), tolerance=0.1)
        bgmask = bgmask.astype(int)
        wingmask=flood(bgmask, (fw_y0,fw_x0), tolerance=0.1)
        fw_1_im[wingmask==False]=(255,255,255)

        bgmask = flood(fw_0_im[..., 1], (1,1), tolerance=0.1)
        bgmask = bgmask.astype(int)
        wingmask=flood(bgmask, (fw_y0,fw_x0), tolerance=0.1)
        fw_0_im[wingmask==False]=(255,255,255)

        #CROP
        crop_wing_func(fw_0_im,fw_1_im, file_save_crop,sorted_file_list_full[4*i],sorted_file_list_full[4*i+2])
        crop_wing_func(hw_0_im,hw_1_im, file_save_crop,sorted_file_list_full[4*i+1],sorted_file_list_full[4*i+3])


      if file_category=="sticky":
        #READ DATA
        fw_0_im=cv2.imread(file_path_read + sorted_file_list_full[4*i],-1)
        hw_0_im=cv2.imread(file_path_read + sorted_file_list_full[4*i+1],-1)
        fw_1_im=cv2.imread(file_path_read + sorted_file_list_full[4*i+2],-1)
        hw_1_im=cv2.imread(file_path_read + sorted_file_list_full[4*i+3],-1)
        #CROP
        crop_wing_func(fw_0_im,fw_1_im, file_save_crop,sorted_file_list_full[4*i],sorted_file_list_full[4*i+2])
        crop_wing_func(hw_0_im,hw_1_im, file_save_crop,sorted_file_list_full[4*i+1],sorted_file_list_full[4*i+3])














