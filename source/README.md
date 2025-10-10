# arphia_conspersa
This folder contains the source code to run each step separately. 

## Structure 


```
source/
    └── 01_find_pt/
    	└── find_pt.py
    └── 02_extraction/
    	└── extraction.py
    └── 03_svd/
    └── 04_segmentation/
    └── 05_venation_network/
    
```

## Run step by step and validations 
cd source 
### 01. Find points on the wing  
Goal: identify different regions (background, forewing, hindwing, body) on the wing to help extraction.<br>
Input: raw image 
```bash
python ./01_find_pt/find_pt.py
```
Output: positions of pt on the wing. <br>
Validations: check if the points on the wing (an example). 

### 02. Wing extraction 
Goal: Seperate background, forewing, hindwing.<br>
Input: pt and raw image 
```bash
python extraction.py
```
Output: seperated forewing and hindwing.<br>
Validations: 

### 03. Alignment and svd
Goal: Align two images and do svd that emphasis on the wing skeleton. <br>
Input: hindwing in transmitted light and reflected light
```bash
bash ./03_svd/align_img/align_img.sh
python ./03_svd/svd_img.py
```
Output: grey image <br>
Validations: check the alignment results. Check the svd results. 

### 04. Domain segmentation 
Goal: segment domains and veins from the hindwing using svd image. <br>
Input: grey image
```bash
python segmentation.py
```
Output: masks of domains. <br>
Validations: check if domains are correctly identified. If not, cellpose allows manual correction and retraining the model. 

### 05. Venation network 
Goal: convert the masks into a graph, with vertices and edge (contain thickness). <br>
Input: hindwing image, masks of domains.
```bash
python venation_network.py
```
Output: graph, binary image. 
