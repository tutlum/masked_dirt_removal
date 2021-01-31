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
from gimpfu import *
import math
import os
import multiprocessing as mp
import numpy as np
from functools import partial

BRUCH_HARDNESS=0.9
RING_SIZE_MULTIPLICATION=1.2
CONTRAST_RAD_MULTIPL=3

stdcoord=[np.array([math.cos(w*math.pi/3),math.sin(w*math.pi/3)]) for w in range(6)]

def channelData(layer):
    w,h=layer.width,layer.height
    region=layer.get_pixel_rgn(0, 0, w,h)
    pixChars=region[0:w,0:h]
    bpp=region.bpp
    return np.frombuffer(pixChars,dtype=np.uint8).reshape(h,w,bpp)

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
    
    neighbors=[(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1)]
    
    
    while stack:
        v=stack.pop()
        if arr[v]==-1:
            arr[v]=cl
            cl_arr[cl].append(v)
            for (i,j) in neighbors:
                if (0<=v[0]+i<size[0] and 0<=v[1]+j<size[1] and arr[v[0]+i,v[1]+j]==-1):
                    stack.append((v[0]+i,v[1]+j))
    
def approx_make_circle(data):
    arr=np.array(data)
    mina=arr.min(0)
    maxa=arr.max(0)
    center=(mina+maxa)/2
    radius=max(maxa-center)
    return center[0],center[1],radius

def calc_circles(mask, ignore_px_thr=2):
    size=mask.shape
    print(size)
    cl_m={}
    circles=[]
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
            #circles.append(sec.make_circle(cl_m[m]))
    
    return np.array(circles)
    

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
    return ((maxi[0]-mini[0])/(aver[0]+0.01)+(maxi[1]-mini[1])/(aver[1]+0.01)+(maxi[2]-mini[2])/(aver[2]+0.01))/3.0, (sum(aver)/255.0)/3.0


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
    return midpoints[np.argmin(sample_values)]
        
#just for history or perhaps bigger images 
def get_heal_data2(coords, layerdata, sel_radius, contrast_thr=1.3,brightness_thr=0.08, rings=1):
    data=[]
    for j,(x,y,r) in enumerate(coords):
        sc=samplecontrast(get_original_samples(layerdata, x, y, CONTRAST_RAD_MULTIPL*RING_SIZE_MULTIPLICATION*r))
        
        if sc[0]<contrast_thr and sc[1]>brightness_thr:
            src = sample_around(layerdata, x, y, RING_SIZE_MULTIPLICATION*sel_radius*r,rings=rings)
            data.append(((x,y,sel_radius*r),src))
    return data
    
def get_heal_data(coord, layerdata, sel_radius, contrast_thr=1.3,brightness_thr=0.08, rings=1):
    (x,y,r)=coord
    sc=samplecontrast(get_original_samples(layerdata, x, y, CONTRAST_RAD_MULTIPL*RING_SIZE_MULTIPLICATION*r))
    
    if sc[0]<contrast_thr and sc[1]>brightness_thr:
        #print("{}\t{}".format(sc,(x,y)))
        src = sample_around(layerdata, x, y, RING_SIZE_MULTIPLICATION*sel_radius*r,rings=rings)
        return ((x,y,sel_radius*r),src)
    #else:
        #print("{}\t{}\tX".format(sc,(x,y)))
    return None
    
def execute_heal_data(layer, coords):
    size=len(coords)
    pdb.gimp_context_set_brush("2. Hardness 100")
    pdb.gimp_context_set_brush_hardness(BRUCH_HARDNESS)
    for j,((x,y,r),(src_x,src_y)) in enumerate(coords):
        pdb.gimp_context_set_brush_size(2*r)
        pdb.gimp_heal(layer, layer, src_y, src_x, 2, [y,x])
        if j%10==0:
            pdb.gimp_progress_set_text("Healed {} of {} wounds".format(j,size))
            pdb.gimp_progress_update(float(j)/size)

def init(image, layer, sample_points=6, mask_path=""):
    import sys
    sample_points=int(sample_points)
    read=False # Flag if mask was read by mask_path
    circles=None
    
    if sample_points!=6:
        stdcoord=[np.array([math.cos(w*2*math.pi/sample_points),math.sin(w*2*math.pi/sample_points)]) for w in range(sample_points)]
    
    start_t=time.time()
    if os.path.isfile(mask_path) and mask_path.endswith(".npz"):
        dict_data = np.load(mask_path)
        circles = dict_data['arr_0']
        if circles.shape[1]!=3:
            raise ValueError("Not a valid mask")
        read=True
    elif mask_path!="":
        raise ValueError("Not a usable file for a mask (only .npz files).")
    
    if not read:
        dirt_layer = pdb.gimp_image_get_layer_by_name(image,"Dirt Mask")
        if dirt_layer is None:
            pdb.gimp_message("You must provide a .npz file as mask or generate one by first choosing a layer named 'Dirt Mask' as heal-template (dark is dirt, white is clean)")
            return
    
    src_img=layer
    src_data=channelData(src_img)
    
    if not read:
        arr=channelData(dirt_layer)
        arr1=arr[:,:,0]
        mask=np.array(np.where(arr1<128,-1,0))
    #arr2=np.array(np.where(arr1<128,1,0))
    
    print("load time: ", time.time()-start_t)
    
    
    if not read:    
        start_t=time.time()
        circles = calc_circles(mask)
        print("total {} circles".format(len(circles)))
        print("region time: ", time.time()-start_t)
        mp.process
    
    return src_img, src_data, circles, read


# saving the calculated circles
def save_circles(read, circles):
    if not read:
        try:
            import tkFileDialog
            f=tkFileDialog.asksaveasfile(mode='w', defaultextension=".npz")
            if not f is None:
                np.savez_compressed(f,circles)
                f.close()
        except Exception:
            print("No TK")


def heal_image(image, layer, sel_radius=1.1, sample_points=6, contrast_thr=1.3,brightness_thr=0.08,mask_path=""):
    import sys
    src_img, src_data, circles, read = init(image, layer, sample_points, mask_path)
    
    pdb.gimp_image_undo_group_start(image)
    pdb.gimp_context_push()
    
    
    results=[]
    pdb.gimp_progress_set_text("Calculating Wounds")
    pdb.gimp_progress_update(0.01)
    start_t=time.time()
    #pool=mp.Pool()
    #results=pool.map(partial(get_heal_data2,layerdata=src_data, sel_radius=sel_radius, contrast_thr=contrast_thr,brightness_thr=brightness_thr),np.array_split(circles, mp.cpu_count()))
    results=map(partial(get_heal_data,layerdata=src_data, sel_radius=sel_radius, contrast_thr=contrast_thr,brightness_thr=brightness_thr),circles.tolist()) #actually faster than pool.map
    print("find heal time: ", time.time()-start_t)
    
    print("Found wounds:", len([r for r in results if not r is None]))
    
    start_t=time.time()
    #for r in results:
    #    execute_heal_data(src_img, r)
    
    execute_heal_data(src_img, [r for r in results if not r is None])
    print("heal time: ", time.time()-start_t)
    
    # for i,(x,y,r) in enumerate(circles):
        
        # heal(image, src_img, src_data, x, y, sel_radius*r, contrast_thr, brightness_thr, rings=1)
        # if i%10==0:
            # pdb.gimp_progress_set_text("Healed {} of {} particles".format(i,numcirc))
            # pdb.gimp_progress_update(float(i)/numcirc)
    # print "heal time: ", time.time()-start_t
        
        
    pdb.gimp_context_pop()
    pdb.gimp_image_undo_group_end(image)
    pdb.gimp_displays_flush()

    save_circles(read, circles)

  
register(
	"remove_dirt",                           
	"Remove Dirt",
	"Remove Dirt: Select a layer, with dirt on it and name a Layer 'Dirt Mask' where black is dirt, white is clean",
	"Sascha Schleef",
	"Sascha Schleef",
	"July 2020",
	"<Image>/Python-Fu/Remove Dirt",             #Menu path
	"RGB*, GRAY*", 
	[
    (PF_SPINNER, "sel_radius", "Factor Selection:", 2.5, (0, 3, 0.1)),
    #(PF_SPINNER, "rings", "Rings of Testing around center:", 1, (1, 3, 1)),
    (PF_SPINNER, "sample_points", "Number of testing points on each ring:", 42, (5, 50, 1)),
    (PF_SLIDER, "contrast_thr", "Contrast-Grenze:", 1.3, (0, 5, 0.05)),
    (PF_SLIDER, "brightness_thr", "Helligkeitsgrenze:", 0.08, (0, 1, 0.01)),
    (PF_FILENAME, "mask_path", "Path to mask file:","")
    ],
	[],
	heal_image)
    
main()

