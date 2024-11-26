# write name_list.txt for a population folder 
import os
path="./imgs/population_173"
dir_list=os.listdir(path)
filename=path+"/name_list.txt"
f=open(filename,"w")
for o in dir_list:
    if o[0]=="p" and o[34]=="0":
        f.write(o[:27]+"\n")
        print(o)
f.close()
# print(dir_list)