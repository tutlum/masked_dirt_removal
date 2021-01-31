#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  remove_dirt.py
#  
#  Copyright 2020 Sascha Schleef <sschleef@homebody>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import sys
print(sys.executable)

import time
# Can be used for more acurate but slower calculation of minimum enclosing circle
#import smallestenclosingcircle as sec
import math
import os
import multiprocessing as mp
import numpy as np
from functools import partial
from PIL import Image

BRUCH_HARDNESS=0.9
RING_SIZE_MULTIPLICATION=1.2
CONTRAST_RAD_MULTIPL=3

stdcoord=[np.array([math.cos(w*math.pi/3),math.sin(w*math.pi/3)]) for w in range(6)]

neighbors=[(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1)]

def createResultLayer(image,name,result):
    rlBytes=np.uint8(result).tobytes();
    rl=gimp.Layer(image,name,image.width,image.height,
                  image.active_layer.type,100,NORMAL_MODE)
    region=rl.get_pixel_rgn(0, 0, rl.width,rl.height,True)
    region[:,:]=rlBytes
    image.add_layer(rl,0)
    gimp.displays_flush()
    return rl

def mk_cluster(arr,x,y, size, cl, cl_arr):
    stack=[(x,y)]
    cl_arr[cl].append((x,y))
    
    
    while stack:
        v=stack.pop()
        if arr[v]==-1:
            arr[v]=cl
            cl_arr[cl].append(v)
            for (i,j) in neighbors:
                if (0<=v[0]+i<size[0] and 0<=v[1]+j<size[1] and arr[v[0]+i,v[1]+j]==-1):
                    stack.append((v[0]+i,v[1]+j))

def get_clustermask(mask_data,cluster_points):
    arr=np.array(cluster_points)
    mina=arr.min(0)
    maxa=arr.max(0)
    return mask_data[mina[0]:maxa[0]+1,mina[1]:maxa[1]+1]

def approx_make_circle(data):
    arr=np.array(data)
    mina=arr.min(0)
    maxa=arr.max(0)
    center=(mina+maxa)/2
    radius=max(maxa-center)
    return int(center[0]),int(center[1]),int(radius+1)

def calc_circles(mask, ignore_px_thr=2,mask_data=None):
    size=mask.shape
    #print(size)
    cl_m={}
    circles=[]
    circle_masks=[]
    cl=0
    for i in range(size[0]):
        for j in range(size[1]):
            if (mask[i,j]==-1):
                cl+=1
                cl_m[cl]=[]
                mk_cluster(mask,i,j,size,cl,cl_m)
    
    for m in cl_m:
        if len(cl_m[m])>ignore_px_thr:
            circles.append(approx_make_circle(cl_m[m]))
            circle_masks.append(get_clustermask(mask_data,cl_m[m]))
            #circles.append(sec.make_circle(cl_m[m]))
    
    return np.array(circles),circle_masks
    

def coord_in_range(coord,maxx,maxy):
    if 0<coord[0]<maxx and 0<coord[1]<maxy:
        return coord
    else:
        return False

def samplecontrast(samples):
    maxi=[0.0,0.0,0.0]
    mini=[0.0,0.0,0.0]
    aver=[0.0,0.0,0.0]
    for col in samples:
        for i in range(3):
            aver[i]+=col[i]
            if col[i]>maxi[i]: maxi[i]=col[i]
            if col[i]<mini[i]: mini[i]=col[i]
    for i in range(3): aver[i]=aver[i]/len(samples)
    return sum([(maxi[i]-mini[i])/(aver[i]+0.01) for i in [0,1,2]])/3.0, (sum(aver)/255.0)/3.0


def get_original_samples(layerdata, x, y, rad):
    size=layerdata.shape
    orig_samples=[]
    orig=np.array([x,y])
    prevcoor=(0,0)
    for coord in stdcoord:
        c=coord_in_range(tuple((orig+coord*rad).astype(int)),size[0],size[1])
        if c:
            orig_samples.append(layerdata[c])
            prevcoor=c
        else:
            orig_samples.append(layerdata[prevcoor])
    return orig_samples



# Todo: Double Code
def sample_around(layerdata, x, y, rad, rings=1):
    size=layerdata.shape
    orig=np.array([x,y])
    orig_samples=[]
    midpoints=[]
    samples=[]
    
    prevcoor=(0,0)
    
    for run in range(1,rings+1):
        prevcoor=(0,0)
        for coord in stdcoord:
            c=coord_in_range(tuple((orig+coord*rad*run).astype(int)),size[0],size[1])
            if c:
                orig_samples.append(layerdata[c])
                prevcoor=c
                for coord_s in stdcoord:
                    midpoint=orig+coord_s*2*rad
                    if coord_in_range(tuple(midpoint.astype(int)),size[0],size[1]):
                        midpoints.append(midpoint)
                        sample_samples=[]
                        for run2 in range(1,rings+1):
                            for coord_ss in stdcoord:
                                c=coord_in_range(tuple((midpoint+coord_ss*rad*run2).astype(int)),size[0],size[1])
                                if c:
                                    sample_samples.append(layerdata[c])
                                    prevcoor=c
                                else:
                                    sample_samples.append(layerdata[prevcoor])
                        samples.append(sample_samples)
            else:
                orig_samples.append(layerdata[prevcoor])
    
    orig_samples_np=np.tile(np.array(orig_samples),[len(samples),1,1])
    data=np.array(samples)
    
    #calculates minimal color differences(pointwise):
    sample_values=np.sum(np.square(data-orig_samples_np),axis=(1,2))
    return midpoints[np.argmin(sample_values)].astype(int)
        
#just for history or perhaps bigger images 
def get_heal_data2(coords, layerdata, sel_radius, contrast_thr=1.3,brightness_thr=0.08, rings=1):
    data=[]
    for j,(x,y,r) in enumerate(coords):
        data.append(get_heal_data((x,y,r), layerdata, sel_radius, contrast_thr,brightness_thr, rings))
    return data
    
def get_heal_data(coord, layerdata, sel_radius, contrast_thr=1.3,brightness_thr=0.08, rings=1):
    (x,y,r)=coord
    sc=samplecontrast(get_original_samples(layerdata, x, y, CONTRAST_RAD_MULTIPL*RING_SIZE_MULTIPLICATION*r))
    
    if sc[0]>contrast_thr:
        print("Controst high: {},{}".format(x,y)) 
    elif sc[1]<brightness_thr:
        print("Brightness low: {},{}".format(x,y)) 
    else:
        #print("{}\t{}".format(sc,(x,y)))
        src = sample_around(layerdata, x, y, RING_SIZE_MULTIPLICATION*sel_radius*r,rings=rings)
        return ((x,y,sel_radius*r),src)
    #else:
        #print("{}\t{}\tX".format(sc,(x,y)))
    return None
    
def execute_heal_data(image, coords, circle_masks):
    #print(coords)
    size=len(coords)
    over=np.zeros((image.shape[0],image.shape[1],4),np.uint8)
    for j,pair in enumerate(coords):
        if pair is not None:
            (x,y,r),(src_x,src_y)=pair
            w,h=circle_masks[j].shape
            w2,h2=(math.floor(w/2.0),math.ceil(w/2.0)),(math.floor(h/2.0),math.ceil(h/2.0))
            # add mask as alpha channel to mask region defined by w and h TODO: check circle_masks-data
            try:
                over_tmp=np.concatenate((image[src_x-w2[0]:src_x+w2[1],src_y-h2[0]:src_y+h2[1]],circle_masks[j].reshape((w,h,1))),axis=2)
            except ValueError:
                print("out of range: src: x,y+-({},{}) for ({},{})".format(str(w2),str(h2),x,y))
                continue
            try:
                over[x-w2[0]:x+w2[1],y-h2[0]:y+h2[1]]=over_tmp
            except ValueError:
                print("out of range: x,y+-({},{}) for ({},{})".format(str(w2),str(h2),src_x,src_y))
                continue
    #alphaim=np.concatenate((image,np.full((image.shape[0],image.shape[1],1),255)),axis=2)
    img=Image.alpha_composite(Image.fromarray(image,"RGB").convert('RGBA'), Image.fromarray(over,"RGBA"))
    return img.convert('RGB')
            

def init(image, dirt, sample_points=6, mask_path=""):
    import sys
    sample_points=int(sample_points)
    read=False # Flag if mask was read by mask_path
    circles=None
    circle_masks=None
    
    if sample_points!=6:
        global stdcoord
        stdcoord=[np.array([math.cos(w*2*math.pi/sample_points),math.sin(w*2*math.pi/sample_points)]) for w in range(sample_points)]
    print(sample_points)
    
    start_t=time.time()
    if os.path.isfile(mask_path) and mask_path.endswith(".npz"):
        dict_data = np.load(mask_path,allow_pickle=True)
        circles, circle_masks = dict_data['circ'], dict_data['mask']
        if circles.shape[1]!=3:
            raise ValueError("Not a valid mask")
        read=True
    elif mask_path!="" and mask_path!="-":
        raise ValueError("Not a usable file for a mask (only .npz files).")
    
    src_data=np.asarray(Image.open(image).convert("RGB"), np.uint8)
    
    if not read:
        mask_data = np.asarray(Image.open(dirt), np.uint8)
        if mask_data is None:
            print("You must provide a .npz file as mask or generate one by first choosing a layer named 'Dirt Mask' as heal-template (dark is dirt, white is clean)")
            return
        if len(mask_data.shape)!=2:
            mask_data=mask_data[:,:,0]
        if (np.max(mask_data)==1):
            mask_data=mask_data*255
        mask_data=255-mask_data
        mask=np.array(np.where(mask_data>0,-1,0))
    #arr2=np.array(np.where(arr1<128,1,0))
    
    print("load time: ", time.time()-start_t)
    
    
    if not read:    
        start_t=time.time()
        #print_img(mask_data)
        circles,circle_masks = calc_circles(mask,mask_data=mask_data)
        print("total {} circles".format(len(circles)))
        print("region time: ", time.time()-start_t)
        mp.process
    
    return src_data, circles, circle_masks, read


def print_img(img):
    for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                print(print_pxl(int(img[i,j]*4.0/255.0)),end='\n' if j==img.shape[1]-1 else '')
def print_pxl(pxl):
    if pxl==0: return "."
    elif pxl==1: return ":"
    elif pxl==2: return "o"
    elif pxl==3: return "O"
    elif pxl==4: return "0"

# saving the calculated circles
def save_circles(read, circles, circle_masks, filename="masks.npz"):
    if not read:
        fn=input("Filename for storing masks ({}): ".format(filename))
        if fn == "": fn=filename
        with open(fn, "wb") as f:
            np.savez_compressed(f,circ=circles,mask=circle_masks)


def heal_image(image, dirt, mask_path="masks.npz", sel_radius=1.1, sample_points=6, contrast_thr=1.3,brightness_thr=0.08):
    import sys
    src_data, circles, circle_masks, read = init(image, dirt, sample_points, mask_path)
    results=[]
    #print(circles)
    start_t=time.time()
    pool=mp.Pool()
    result=pool.map(partial(get_heal_data2,layerdata=src_data, sel_radius=sel_radius, contrast_thr=contrast_thr,brightness_thr=brightness_thr),np.array_split(circles, mp.cpu_count()))
    results=[]
    for l in result: results.extend(l)
    #results=list(map(partial(get_heal_data,layerdata=src_data, sel_radius=sel_radius, contrast_thr=contrast_thr,brightness_thr=brightness_thr),circles.tolist())) #actually faster than pool.map
    print("find heal time: ", time.time()-start_t)
    
    #Image.fromarray(src_data,"RGB").save("testout_src.jpg") 
    print("Found wounds:", len([r for r in results if not r is None]))
    
    start_t=time.time()
    #for r in results:
    #    execute_heal_data(src_img, r)
    
    im=execute_heal_data(src_data, results, circle_masks)
    im.save("testout.jpg") 
    print("heal time: ", time.time()-start_t)
    
    # for i,(x,y,r) in enumerate(circles):
        
        # heal(image, src_img, src_data, x, y, sel_radius*r, contrast_thr, brightness_thr, rings=1)
        # if i%10==0:
            # pdb.gimp_progress_set_text("Healed {} of {} particles".format(i,numcirc))
            # pdb.gimp_progress_update(float(i)/numcirc)
    # print "heal time: ", time.time()-start_t
        

    save_circles(read, circles, circle_masks, filename=mask_path)

    
    
if __name__ == "__main__":
    print(sys.argv[1:])
    heal_image(*sys.argv[1:],contrast_thr=2,sample_points=30)

