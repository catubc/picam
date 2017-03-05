#Support functions for picam code

import os, sys, csv
import numpy as np
import matplotlib.pyplot as plt
import parmap
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.path import Path
import scipy
import scipy.ndimage
import scipy.signal
from mpl_toolkits.mplot3d import axes3d
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

import skimage
from skimage import data
from skimage.transform import rotate
#from scipy.signal import butter, filtfilt, cheby1

sys.path.append('/home/cat/code/')
import TSF.TSF as TSF
import PTCS.PTCS as PTCS


def parallel_divide(imaging, baseline):
    return np.divide(imaging-baseline, baseline)

def find_nearest_180(array,value,dist):
    return_stack = []
    for k in range(-90,90,1):
        index = np.abs(array-(value+(k*dist))).argmin()
        if abs(array[index]-(value+(k*dist))) < dist:
            return_stack.append(index)
        else:
            return_stack.append(None)

        #print k, value+(k*dist), index, array[index], return_stack[k+90]
    
    return return_stack

def find_nearest_180_parallel(value, array, dist):
    return_stack = []
    for k in range(-90,90,1):
        index = np.abs(array-(value+(k*dist))).argmin()
        if abs(array[index]-(value+(k*dist))) < dist:
            return_stack.append(index)
        else:
            return_stack.append(None)

        #print k, value+(k*dist), index, array[index], return_stack[k+90]
    
    return return_stack

def find_ephys_epochs(P): #lfp_filename, selected_epoch):
    
    epoch_file = os.path.split(P.lfp_filename)[0]+"/ephys_epochs.txt"
    if os.path.exists(epoch_file)==False: 

        tsf = TSF.TSF(P.lfp_filename)
        tsf.read_footer()

        on_times = []
        off_times = []
        offset = 0

        for f in range(tsf.n_files[0]):
            #for k in range(tsf.n_digital_chs[f]):
            k = 0 
            plt.plot(tsf.digital_chs[f][k][::10])

            #Find transitions to on and off
            for s in range(0, len(tsf.digital_chs[f][k])-25, 25):
                if tsf.digital_chs[f][k][s]!=tsf.digital_chs[f][k][s+25]:
                    if tsf.digital_chs[f][k][s]==0: on_times.append(s+offset)
                    else: off_times.append(s+offset)
            plt.show()
            offset = offset + tsf.n_samples[f] #Must offset future times by recording time;
            
        ephys_epochs = []
        for k in range(len(on_times)):
            ephys_epochs.append([on_times[k], off_times[k]])
        print ephys_epochs

        np.savetxt(epoch_file, ephys_epochs, fmt='%i')

    else:
        ephys_epochs = np.loadtxt(epoch_file)
        
    return ephys_epochs[P.selected_epoch]

def rotate_imaging(sta_array, sua_filename):
    
    root_dir = os.path.split(os.path.split(sua_filename)[0])[0]
    n_pixels = int(np.loadtxt(root_dir+'/n_pixels.txt'))
    
    image_shift_file = sua_filename[:-5]+"_image_shift.txt"

    sta_copy = np.float64(sta_array.copy())

    #Recenter/shift images:
    
    print "Rotating images"
    with open(image_shift_file, "r") as f: #Text file contains image shift and rotation angle info
        data = csv.reader(f)
        temp = []
        for row in data:
            temp.append(int(row[0]))
    x_shift = temp[0]
    y_shift = temp[1]
    angle = temp[2]
    
    print x_shift, y_shift, angle
    
    #*******************SHIFT IMAGE******************
   
    if False: 


        midline_mask = 0
        main_dir = os.path.split(os.path.split(sua_filename)[0])[0]
        sta_copy = mask_data(sta_copy, main_dir, midline_mask, sua_filename)
        sta_copy = np.float64(sta_copy)

        ax = plt.subplot(221)
        plt.imshow(sta_copy[90], vmin = -0.10, vmax=0.10)

        ax = plt.subplot(222)
        plot_2 = np.roll(sta_copy[90], y_shift, axis=0)
        plot_2[n_pixels/2] = 0
        plot_2[:, n_pixels/2] = 0
        plt.imshow(plot_2, vmin = -0.10, vmax=0.10)
        
        ax = plt.subplot(223)
        plot_3 = np.roll(sta_copy[90], x_shift, axis=1)
        plot_3 = np.roll(plot_3, y_shift, axis=0)
        plot_3[n_pixels/2] = 0
        plot_3[:, n_pixels/2] = 0
        plt.imshow(plot_3, vmin = -0.10, vmax=0.10)
        
        ax = plt.subplot(224)
        plot_4 = skimage.transform.rotate(sta_copy[90], angle)#, mode='constant', cval=100)
        plot_4 = np.roll(plot_4, x_shift, axis=1)
        plot_4 = np.roll(plot_4, y_shift, axis=0)
        plot_4[n_pixels/2] = 0
        plot_4[:, n_pixels/2] = 0
        plt.imshow(plot_4, vmin = -0.10, vmax=0.10)

        plt.show()


    #*******************ROTATE IMAGE********************
    print "Rotate angle: ", angle
    
    sta_array = np.float64(sta_array)

    if angle != 0:

        for i in range(len(sta_copy)):
            sta_copy[i] = skimage.transform.rotate(sta_copy[i], angle)#, mode='constant', cval=100)

            sta_copy[i] = np.roll(sta_copy[i], x_shift, axis=1)
            sta_copy[i] = np.roll(sta_copy[i], y_shift, axis=0)


    return sta_copy
    
    

def filter_imaging(P):
    
    print "...loading imaging epochs..."
    root_dir = os.path.split(os.path.split(P.imaging_files[0])[0])[0]
    n_pixels = int(np.loadtxt(root_dir+'/n_pixels.txt'))

    #LOAD Imaging data for specific epoch
    #for ctr, filename in enumerate(imaging_files):
    #    if ctr!=selected_epoch: continue
    filename = P.imaging_files[P.selected_epoch]
    if os.path.exists(filename[:-4]+".npy")==False:
        
        print "... npy files do not exist: making and saving..."
        
        stream = open(filename, 'rb')
        
        #Y = np.fromfile(stream, dtype=np.uint8, count=width*height*frames*3).reshape((frames, height, width, 3))
        Y = np.fromfile(stream, dtype=np.uint8).reshape((-1, n_pixels, n_pixels, 3))
        
        
        np.save(filename[:-4], Y)
        
    else:
        Y = np.load(filename[:-4]+'.npy', mmap_mode='c')
            
    
    ###LOAD MASK - filter only datapoints inside mask
    #main_dir = os.path.split(os.path.split(sua_filename)[0])[0]
    #generic_mask_file = main_dir + '/genericmask.txt' 
    #if os.path.exists(generic_mask_file)==False:
        #Define_generic_mask(np.array(data), main_dir)
        
    #generic_coords = np.loadtxt(generic_mask_file)
    #generic_mask_indexes=np.zeros((n_pixels,n_pixels))
    #for i in range(len(generic_coords)): 
        #generic_mask_indexes[int(generic_coords[i][0])][int(generic_coords[i][1])] = True
    
    
    #**************** SELECT GREEN CHANNEL ONLY ********************
    Y = Y[:,:,:,1]     
    P.imaging_original = Y[:,::-1,::-1]

    
    #**************** COMPUTE MEAN OF STACK ************************
    
    if os.path.exists(filename[:-4]+"_epoch"+str(P.selected_epoch)+"_mean.npy")==False:
        print "... computing imaging mean..."
        Y_mean = np.mean(Y, axis=0)
        
        np.save(filename[:-4]+"_epoch"+str(P.selected_epoch)+"_mean", Y_mean)
    else:
        
        Y_mean = np.load(filename[:-4]+"_epoch"+str(P.selected_epoch)+"_mean.npy")
        
    
    #****************** FILTER *********************************
    if os.path.exists(filename[:-4]+"_green_filtered.npy")==False:
        
        filtered_stack=[]
        
        print "... making filters, and lists of pixels..."
        cutoff = 0.1
        order = 4
        SampleFrequency = 30
        temp_traces = []

        nyq = 0.5 * SampleFrequency
        normal_cutoff = cutoff/nyq
        b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
        
        print Y.shape
        
        pixel_array = []
        for i in range(n_pixels):
            for j in range(n_pixels):
                pixel_array.append(Y[:,i,j])
        
        #Use parmap
        import parmap
        
        print "... highpass filtering data (parallel version)..."
        filtered_pixels = parmap.map(do_filter, pixel_array, b, a, processes=P.n_processes)
        

        print "... reconstructing data ..."
        ctr=0
        Y_filtered = np.zeros(Y.shape, dtype=np.float32)
        for p1 in range(n_pixels):
            print "...row: ", p1
            for p2 in range(n_pixels):
                #if generic_mask_indexes[p1,p2]==False:
                Y_filtered[:,p1,p2] = filtered_pixels[ctr]; ctr+=1
        
        print "...  saving data..."
        np.save(filename[:-4]+'_green_filtered', Y_filtered)



    else:
        
        Y_filtered = np.load(filename[:-4]+'_green_filtered.npy', mmap_mode='c')

    P.imaging_filtered = Y_filtered[:,::-1,::-1]     #THIS INVERTS THE DATA; IT IS SAVED RIGHT SIDE UP THOUGH
    #return Y_filtered[:,::-1,::-1]     #THIS INVERTS THE DATA; IT IS SAVED RIGHT SIDE UP THOUGH

           
def do_filter(pixel, b, a):
    return np.float16(scipy.signal.filtfilt(b, a, pixel))



def set_blue_light(P, imaging_epoch):
    print "...setting blue light..."
  
    root_dir = os.path.split(os.path.split(P.imaging_files[P.selected_epoch])[0])[0]
    n_pixels = int(np.loadtxt(root_dir+'/n_pixels.txt'))
    imaging_onoff_file = P.imaging_files[P.selected_epoch][:-4]+'_imaging_onoff.txt'        #This sets the imaging_onoff for each recording in the track

    if os.path.exists(imaging_onoff_file)==False: 
        
        Y = np.load(P.imaging_files[P.selected_epoch][:-4]+'.npy', mmap_mode='c')
        
        imaging_epoch = Y[:,:,:,1]     
        
        #intensity = np.mean(imaging_epoch[:1000].reshape(1000,65536), axis=1)
        #plt.plot(intensity)
        #plt.show()

        blue = imaging_epoch[:,int(n_pixels/2):int(n_pixels/2)+1,n_pixels/2]
        
        lighton_trace = np.mean(blue, axis=1)

        print ">>>>> make file: \n\n", imaging_onoff_file, "\n\n  <<<<<  and SAVE ONOFF TIMES"

        plt.plot(lighton_trace)
        plt.show()

    P.imaging_onoff = np.loadtxt(imaging_onoff_file, dtype=np.int32)

    P.imaging_filtered_lighton = imaging_epoch[P.imaging_onoff[0]:P.imaging_onoff[1]]

    #return imaging_onoff, imaging_epoch
    

def make_stack_parallel(frames, dff_stack):
    
    #sta_array=[]
    #print "...plotting frames..."
    #for ctr,frames in enumerate(frame_stack):               #********************************************PARALLELIZE THIS!!!!
    #ax = plt.subplot(10,18,ctr+1)
    frames_temp = []
    for k in frames:
        if k != None: 
            frames_temp.append(k)
    temp_ave = np.mean(dff_stack[np.int32(frames_temp)],axis=0)

    return temp_ave 
    
    
    sta_array=[]
    print "...generating frames..."
    for ctr,frames in enumerate(frame_stack):               #********************************************PARALLELIZE THIS!!!!
        #ax = plt.subplot(10,18,ctr+1)
        frames_temp = []
        for k in frames:
            if k != None: 
                frames_temp.append(k)
        print "...frame: ", ctr, "  # of indexes: ", len(frames_temp)
        temp_ave = np.mean(dff_stack[np.int32(frames_temp)],axis=0)
        sta_array.append(temp_ave)
    




def set_frame_times(P, imaging_epochs, imaging_onoff):
    ''' Clip imaging stack to match on/off times of light
    '''
        
    print "...setting frame times..."

    frame_times = []
    imaging_file = P.imaging_files[P.selected_epoch]
    temp_times = np.loadtxt(imaging_file+"_time.txt",dtype=str)
    if temp_times[0]=="None":
        frame_times= np.int64(temp_times[1:])
    else:
        frame_times = np.int64(temp_times)
    
    print frame_times, len(frame_times)
    print imaging_onoff
    print P.ephys_epochs

    #***************** ASIGN REAL TIME TO IMAGING FRAMES ******************
    all_frametimes = []
    all_imaging = []

    print "* Rec epoch: ", P.selected_epoch
    print "... start ephys: ", P.ephys_epochs[0]/25000.
    print "... duration ephys: ", (P.ephys_epochs[1]-P.ephys_epochs[0])/25000.

    #Check if recording light was properly turned off
    if (imaging_onoff[1] - imaging_onoff[0])<=0:
        print " ****************** error in imaging ****************** \n"
        return

    temp_frametimes = frame_times[imaging_onoff[0]:imaging_onoff[1]]       #Clip frame times to on/off of light
    temp_frametimes = temp_frametimes - temp_frametimes[0]                          #Offset all imgframes to zero start
    temp_frametimes = temp_frametimes*1E-6                                          #Convert frametimes to seconds
    temp_frametimes = temp_frametimes+P.ephys_epochs[0]/25000.                     #Offset imgtimes to ephys trigger

    print "... # img frames: ", len(temp_frametimes)
    print "... duration imaging: ", (temp_frametimes[-1]-temp_frametimes[0])
    print "... frametimes: ", temp_frametimes
    
    #print "missed frames: ", (ephys_epochs[k][1]-ephys_epochs[k][0])/25000.*30 - len(temp_frametimes)

    print '\n'

    all_frametimes.extend(temp_frametimes)
    #all_imaging.append(imaging_epochs[k])
    all_imaging = imaging_epochs

    #Save frametimes for each image and imaging frames at the same time:
    all_frametimes = np.float32(all_frametimes)
    #if os.path.exists(imaging_files[0][:-4]+"_allimaging_frametimes.txt")==False:              #NOT NECESSARY TO SAVE THIS, It's already fast.
    #    np.savetxt(imaging_files[0][:-4]+"_allimaging_frametimes.txt", all_frametimes)

    if True:        #Don't save single epoch imaging again to file - may wish to implement if multiple 
        pass
    else:
        all_imaging_npy_file = P.imaging_files[0][:-4]+"_allimaging"

        if os.path.exists(all_imaging_npy_file+'.npy')==False:
            all_imaging = np.vstack(all_imaging)
            np.save(all_imaging_npy_file, all_imaging)
        else:
            all_imaging = np.load(all_imaging_npy_file+'.npy', mmap_mode='c')

    print all_imaging.shape
    print len(all_frametimes)
    print all_frametimes

    P.all_frame_times = all_frametimes
    P.imaging_filtered_lighton = all_imaging

    P.imaging = all_imaging
    #return all_frametimes, all_imaging



def dff_mean(P):
    
    print "...computing dff..."


    #************** MAKE GREEN CHANNEL
    #imaging = all_imaging[:,:,:,channel]    #Flip UP-DOWN and LEFT-RIGHT while also selecting only Green channel

    stack_dff_filename = P.imaging_files[P.selected_epoch][:-4]+"_green_dffmean"
    if os.path.exists(stack_dff_filename+'.npy')==False:
        
        print "...loading mean..."
        baseline = np.load(P.imaging_files[P.selected_epoch][:-4]+"_epoch"+str(P.selected_epoch)+"_mean.npy")
        
        #baseline = np.mean(np.float32(imaging), axis=0)
        #baseline = np.float32(baseline)
        print baseline.shape
        print baseline[64]
        #plt.imshow(baseline)
        #plt.show()
     
        ##Compute division in parallel
        #if False: 
            #all_imaging_list = []
            #list_indexes = np.int32(np.linspace(0, len(imaging), n_processes+1))
            #print list_indexes
            #for k in range(len(list_indexes[:-1])):                                             #**** DON"T NEED TO DO MAKE LIST FOR PARMAP ****
                #all_imaging_list.append(imaging[list_indexes[k]:list_indexes[k+1]])
                
            #print "...computing division..."

            #dff_stack=[]
            #dff_stack.extend(parmap.map(parallel_divide, all_imaging_list, baseline, processes = n_processes))
            #print "...len parallel list: ", len(dff_stack)
        
            #dff_stack = np.vstack(dff_stack)
            #print dff_stack.shape

        #else:
            
        dff_stack = np.zeros(P.imaging.shape, dtype=np.float32)
        for k in range(len(P.imaging)):
            print "...dividing frame: ", k
            #dff_stack[k]=np.divide(imaging[k]-baseline, baseline)
            dff_stack[k]=np.divide(P.imaging[k], baseline)
            
            #if k%10==0:
                #ax = plt.subplot(1,2,1)
                #plt.imshow(imaging[k])
                #print baseline[64]
                #print imaging[k][64]

                #print dff_stack[k][64]

                #ax = plt.subplot(1,2,2)
                #plt.imshow(dff_stack[k], vmin=-0.15, vmax = 0.15)
                
                #plt.show()
    
        np.save(stack_dff_filename, dff_stack)

    dff_stack = np.load(stack_dff_filename+'.npy', mmap_mode='c')
    print dff_stack.shape
    
    P.dff_stack_green = dff_stack
    P.dff_stack = dff_stack
    
    #return dff_stack



def show_dff_movies(dff_stack_green, sua_filename):

    
    midline_mask = P.P.midline_mask
    main_dir = os.path.split(os.path.split(sua_filename)[0])[0]

    #temp_stack = dff_stack_green[50000:51000]
    temp_stack = dff_stack_green[:1000]
    temp_stack = (temp_stack - np.mean(temp_stack))/np.mean(temp_stack)

    dff_stack_green = mask_data(temp_stack, main_dir, midline_mask, sua_filename, P)
    #dff_stack_blue = mask_data(dff_stack_blue[:1000], main_dir, midline_mask)
    
    fig = plt.figure()
    ax1 = plt.subplot(111)
    plt.title("Green")
    
    #ax4 = plt.subplot(234)
    #ax5 = plt.subplot(235)

    #dff_max = 0.05
    dff_max = np.nanmean(np.abs(dff_stack_green))*2
    dff_min = -dff_max #np.nanmin(np.abs(dff_stack_green))
    print dff_max, dff_min
    im1 = ax1.imshow(dff_stack_green[0], vmin=dff_min, vmax=dff_max)
    
    #vmax = np.nanmax(np.abs(dff_stack_blue)); print vmax
    #im2 = ax2.imshow(dff_stack_blue[0], vmin=-dff_max, vmax=dff_max)

    #vmax = np.nanmax(np.abs(dff_stack_green))
    #im3 = ax3.imshow(dff_stack_blue[0], vmin=0, vmax=10)
    #im4 = ax4.imshow(all_imaging[:,:,:,1][0], vmin=40, vmax=250)
    #im5 = ax5.imshow(all_imaging[:,:,:,2][0], vmin=40, vmax=250)


    def init():
        im1.set_data(np.zeros((128, 128)))
        #im2.set_data(np.zeros((128, 128)))
        #im3.set_data(np.zeros((128, 128)))
        #im4.set_data(np.zeros((128, 128)))
        #im5.set_data(np.zeros((128, 128)))
       

    def animate(i):
        k=i+0
        plt.suptitle("Frame: "+str(k))
        im1.set_data(scipy.ndimage.filters.gaussian_filter(dff_stack_green[k],2))
        #im2.set_data(dff_stack_blue[k])
        #im3.set_data(dff_stack_green[k]/dff_stack_blue[k])

        #im4.set_data(all_imaging[:,:,:,1][k])
        #im5.set_data(all_imaging[:,:,:,2][k])

        #return im1, im2, im3

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(dff_stack_green),
                                   interval=1)

    plt.show()     

    return
               
def on_click_roi(event):
    
    global coords, images_temp, ax, fig, cid, P_temp
    
    n_pix = len(images_temp[0])
    
    if event.inaxes is not None:
        coords.append((event.ydata, event.xdata))
        for j in range(len(coords)):
            for k in range(7):
                for l in range(7):
                    images_temp[int(len(images_temp)/2)][min(n_pix,int(coords[j][0])-1+k)][min(n_pix,int(coords[j][1])-1+l)]=-0.25

        ax.imshow(images_temp[int(len(images_temp)/2)], vmin=-.25, vmax=.25, cmap = P_temp.color_scheme)
        fig.canvas.draw()

    else:
        print 'Exiting'
        plt.close()
        fig.canvas.mpl_disconnect(cid)
        
def Define_roi(images_processed, P):

    global coords, images_temp, ax, fig, cid, P_temp
    
    main_dir = os.path.split(os.path.split(P.sua_filename)[0])[0]

    P_temp = P
    
    images_temp = mask_data(images_processed, P)

    #images_temp = [images_processed.copy()]
    
    print "... Define following ROIs: ", np.loadtxt(main_dir+"/roi_names.txt",dtype=str)
    
    roi_file = main_dir + '/roi_coords.txt'

    if (os.path.exists(roi_file)==False):
        fig, ax = plt.subplots()
        coords=[]

        ax.imshow(images_temp[int(len(images_temp)/2)], vmin=-.15, vmax=.15, cmap=P.color_scheme)#, vmin=0.0, vmax=0.02)
        ax.set_title("Select single pixel to track")
        cid = fig.canvas.mpl_connect('button_press_event', on_click_roi)
        plt.show()

        np.savetxt(roi_file, coords)
        
        P.coords = coords
    else:
        P.coords = np.loadtxt(roi_file, dtype=str)




def mask_data(data, P):
    
    # main_dir, midline_mask, sua_filename, P):
    
    n_pixels = len(data[0])

    main_dir = os.path.split(os.path.split(P.sua_filename)[0])[0]
    
    
    #Load General mask (removes background)
    generic_mask_file = main_dir + '/genericmask.txt' 
    if os.path.exists(generic_mask_file)==False:
        Define_generic_mask(np.array(data), main_dir, P.sua_filename)
           
    generic_coords = np.int32(np.loadtxt(generic_mask_file))
        
    generic_mask_indexes=np.zeros((n_pixels,n_pixels))
    for i in range(len(generic_coords)):
        generic_mask_indexes[generic_coords[i][0]][generic_coords[i][1]] = True

    #Load midline mask
    for i in range(P.midline_mask):
        generic_mask_indexes[:,n_pixels/2+int(P.midline_mask/2)-i]=True
        
    temp_array = np.ma.array(np.zeros((len(data),n_pixels,n_pixels),dtype=np.float32), mask=True)
    #Mask all frames; NB: PROBABLY FASTER METHOD
    for i in range(0, len(data),1):
        temp_array[i] = np.ma.masked_array(data[i], mask=generic_mask_indexes, fill_value = 0)
    
    if P.show_mask:
        
        #Define_roi(sta_array, P)        #Use 92nd frame to draw brainmap;
        coords = np.loadtxt(os.path.split(os.path.split(P.sua_filename)[0])[0]+"/roi_coords.txt")
        n_pix = len(generic_mask_indexes)
        for j in range(len(coords)):
            for k in range(7):
                for l in range(7):
                    generic_mask_indexes[min(n_pix,int(coords[j][0])-1+k)][min(n_pix,int(coords[j][1])-1+l)]=-0.25
      
        plt.imshow(generic_mask_indexes)
        plt.show()
    
    return temp_array



      
def on_click(event):
    
    global coords, img_out, ax, fig, cid, P_temp
    
    n_pix = len(img_out[0])
    
    if event.inaxes is not None:
        coords.append((event.ydata, event.xdata))
        for j in range(len(coords)):
            for k in range(3):
                for l in range(3):
                    img_out[min(n_pix,int(coords[j][0])-1+k)][min(n_pix,int(coords[j][1])-1+l)]=1

        ax.imshow(img_out, vmin=50, vmax=150, cmap=P_temp.color_scheme)
        fig.canvas.draw()

    else:
        print 'Exiting'
        plt.close()
        fig.canvas.mpl_disconnect(cid)


       
def Define_generic_mask(images_processed, P):

    global coords, img_out, ax, fig, cid, P_temp
    
    P_temp = P
    
    main_dir = os.path.split(os.path.split(P.sua_filename)[0])[0]
    
    #ROTATE AND SHIFT IMAGE TO BE DISPLAYED FOR CROPPING FIRST
    image_shift_file = P.sua_filename[:-5]+"_image_shift.txt"
    print "Rotating images"
    with open(image_shift_file, "r") as f: #Text file contains image shift and rotation angle info
        data = csv.reader(f)
        temp = []
        for row in data:
            temp.append(int(row[0]))
    x_shift = temp[0]
    y_shift = temp[1]
    angle = temp[2]
    
    #img_out = skimage.transform.rotate(np.float64(images_processed[int(len(images_processed)/2)]), angle)#, mode='constant', cval=100)
    img_out = skimage.transform.rotate(np.float64(P.imaging_original[int(len(P.imaging_original)/2)]), angle)#, mode='constant', cval=100)
    img_out = np.roll(img_out, x_shift, axis=1)
    img_out = np.roll(img_out, y_shift, axis=0)

    print images_processed.shape

    if (os.path.exists(main_dir + '/genericmask.txt')==False):
        print "...making external mask..."
        fig, ax = plt.subplots()
        coords=[]

        print 
        #ax.imshow(img_out, vmin=-0.1, vmax=0.1, cmap=P.color_scheme)
        ax.imshow(img_out, vmin=50, vmax=150, cmap=P.color_scheme)
        ax.set_title("Compute generic (outside the brain) mask")
        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

        #******* MASK AND DISPLAY AREAS OUTSIDE GENERAL MASK 
        #Search points outside and black them out:
        all_points = []
        for i in range(len(img_out[0])):
            for j in range(len(img_out[0])):
                all_points.append([i,j])

        all_points = np.array(all_points)
        vertixes = np.array(coords) 
        vertixes_path = Path(vertixes)
        
        mask = vertixes_path.contains_points(all_points)
        counter=0
        coords_save=[]
        for i in range(len(img_out[0])):
            for j in range(len(img_out[0])):
                if mask[counter] == False:
                    img_out[i][j]=0
                    coords_save.append([i,j])
                counter+=1

        fig, ax = plt.subplots()
        ax.imshow(img_out, vmin=50, vmax=150, cmap=P.color_scheme)
        plt.show()

        np.savetxt(main_dir+'/genericmask.txt', coords_save)


def compute_sta(unit, P):

#for unit in units:
    print "... cell: ", unit
    
    sta_map_indexes = []
    for k in range(180):
        sta_map_indexes.append([])

    if "50compressed" in P.sua_filename: 
        all_spikes = np.float32(P.sort_sua.units[unit])*1E-6*50.
    else: 
        all_spikes = np.float32(P.sort_sua.units[unit])*1E-6 #/25000.
    print "... no. spikes: ", len(all_spikes)
    
    if len(all_spikes)==0: 
        print "... cell has no spikes..."
        P.epoch_spikes = 0
        return
    
    frame_stack = []
    dt = 0.0333         #Timewindow to search for nearest frame
    #for spike in spikes[:1000]: #use only first 1000 spikes
    #    nearest_frames = find_nearest_180(offset_frame_times, spike, dt)
    #    frame_stack.append(nearest_frames)
    
    n_processes = 25
    

    #************************* EXCLUDE SPIKES OUTSIDE OF IMAGING PERIOD ********************

    spike_indexes = np.where(np.logical_and(all_spikes>=P.all_frame_times[0], all_spikes<=P.all_frame_times[-1]))[0]    #Exclude spikes too close to beginning or end of recordings.
    spikes = all_spikes[spike_indexes]
    
    
    #**************** FIND NEAREST FRAME TIMES FOR EACH SPIKE **************
    
    frame_stack_file = P.sua_filename[:-5]+"_epoch"+str(P.selected_epoch)+"_unit"+str(unit)+"_frame_stack"
    if os.path.exists(frame_stack_file+".npy")==False: 
        print "... finding frame indexes for # spikes: ", len(spikes), ' / ', len(all_spikes)

        frame_stack.append(parmap.map(find_nearest_180_parallel, spikes, P.all_frame_times, dt, processes = n_processes))

        print "... done..."
        frame_stack = np.vstack(frame_stack)
        print frame_stack.shape
        
        frame_stack = frame_stack.T
        np.save(frame_stack_file, frame_stack)
    else:
        frame_stack = np.load(frame_stack_file+'.npy')


    #SAVE ARRAY WITH EVENTS THAT HAVE ALL FRAMES
    if P.make_nopskip_arrays: 
        print "... making no skip arrays..."
        sta_array_noskip_file = P.sua_filename[:-5]+"_epoch"+str(P.selected_epoch)+"_unit"+str(unit)+"_color"+P.stack_color+"_sta_array_noskip"
        if os.path.exists(sta_array_noskip_file+".npy")==False:
            sta_array_noskip = []
            for k in range(len(frame_stack.T)):
                if None in frame_stack.T[k]:
                    pass
                else:
                    sta_array_noskip.append(P.dff_stack[np.int32(frame_stack.T[k])])
            
            print len(frame_stack.T), len(sta_array_noskip)
            
            np.save(sta_array_noskip_file, sta_array_noskip)
        #else:
            #"...loading noskip from disk..."
            #sta_array_no_skip = np.load(sta_array_noskip_file+".npy")

        #quit()

    #********************** COMPUTE STA MAPS **********************

    sta_array_file = P.sua_filename[:-5]+"_epoch"+str(P.selected_epoch)+"_unit"+str(unit)+"_color"+P.stack_color+"_sta_array"
    if os.path.exists(sta_array_file+".npy")==False:
    
        #sta_array.append(parmap.map(make_stack_parallel, frame_stack, dff_stack_green, processes = n_processes))

        sta_array=[]
        print "...generating frames..."
        for ctr,frames in enumerate(frame_stack):               #****************** PARALLELIZE THIS!!!!
            #ax = plt.subplot(10,18,ctr+1)                      
            frames_temp = []
            for k in frames:
                if k != None: 
                    frames_temp.append(k)
            print "...frame: ", ctr, "  # of indexes: ", len(frames_temp)
            
            temp_ave = np.mean(P.dff_stack[np.int32(frames_temp)],axis=0)         #********** ALSO CAN PARALLELIZE THIS - BUT VERY FAST ALREADY

            sta_array.append(temp_ave)
        
        sta_array = np.array(sta_array)
        np.save(sta_array_file, sta_array)
        
    else:
        sta_array = np.load(sta_array_file+'.npy')
    
    print sta_array.shape
    if len(sta_array)==0: 
        print "... cell has no spikes in epoch..."
        P.epoch_spikes = 0
        return
    P.epoch_spikes = len(sta_array)
    print "... spikes in epoch: ", P.epoch_spikes

    #*************************** ROTATE IMAGING STACK ********************

    #Load raw imaging data and save to .npy files; mmap on subsequent loads
    sta_array_rotated = rotate_imaging(sta_array, P.sua_filename)


    #********************* CHECK TO SEE THAT GENERIC MASK HAS BEEN MADE ********************** 
    generic_mask_file = os.path.split(os.path.split(P.sua_filename)[0])[0] + '/genericmask.txt' 
    if os.path.exists(generic_mask_file)==False:
        Define_generic_mask(sta_array_rotated, P)


    #******************** SMOOTH STA ARRAY *******************
    
    for k in range(len(sta_array_rotated)):
        sta_array_rotated[k] = scipy.ndimage.filters.gaussian_filter(sta_array_rotated[k],P.smoothing_pixels)


    #********************* PLOTTING ***********************
    if P.plotting: 
        plot_figure(P.sua_filename, sta_array_rotated, unit, spikes, all_spikes, P.color_scheme, P)
    
    P.green_stack = sta_array_rotated
    P.n_spikes = len(spikes)
    
    #return sta_array_rotated, len(spikes)  #

def show_movies_2by1(unit, P):
    
    midline_mask = P.midline_mask
    
    
    #*********** SMOOTHING
    #if True: 
    #    for k in range(len(P.green_stack)):
    #        green_stack[k] = scipy.ndimage.filters.gaussian_filter(green_stack[k],2)
    
        
    #************ MASK DATA
    P.green_stack = mask_data(P.green_stack,P)
    #blue_stack = mask_data(blue_stack, os.path.split(os.path.split(sua_filename)[0])[0], midline_mask)
    
    
    #***********GENERATE ANIMATIONS
    color_map = P.color_scheme #"jet"

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=3000)

    fig = plt.figure()
    im = []

    #gs = gridspec.GridSpec(2,len(self.ca_stack)*2)
    gs = gridspec.GridSpec(1,2)

    fontsize = 8

    #Green stack
    ax = plt.subplot(gs[0,0])
    ax.get_xaxis().set_visible(False); ax.yaxis.set_ticks([]); ax.yaxis.labelpad = 0
    vmax = np.nanmax(np.abs(P.green_stack))*P.imshow_scaling; vmin=-vmax
    print "...vmax: ", vmax
    #vmax = 0.05; vmin=-vmax
    plt.title("Green ("+str(round(vmin*100,1))+"-"+str(round(vmax*100,1))+"%)", fontsize = fontsize)
    im.append(plt.imshow(P.green_stack[0], vmin=vmin, vmax=vmax,  cmap=color_map, interpolation='none'))
    plt.ylabel("0% centred")
    
    #Green stack - Max dynamics
    ax = plt.subplot(gs[0,1])
    ax.get_xaxis().set_visible(False); ax.yaxis.set_ticks([]); ax.yaxis.labelpad = 0
    vmax = np.nanmax(np.abs(P.green_stack))*0.5; vmin=-vmax #np.nanmin(P.green_stack)
    #vmax = 0.05; vmin=-vmax
    plt.title("Green ("+str(round(vmin*100,1))+"-"+str(round(vmax*100,1))+"%)", fontsize = fontsize)
    im.append(plt.imshow(P.green_stack[0], vmin=vmin, vmax=vmax,  cmap=color_map, interpolation='none'))
    plt.ylabel("'Majid' plots")


    #Loop to combine all video insets into 1
    def updatefig(j):
        print "...animating frame: ", j, ' / ', n_frames
        #plt.suptitle(self.selected_dff_filter+'  ' +self.dff_method + "\nFrame: "+str(j)+"  " +str(format(float(j)/self.img_rate-self.parent.n_sec,'.2f'))+"sec  ", fontsize = 15)
        plt.suptitle("Unit: "+str(unit)+"  # spikes: "+ str(P.n_spikes)+"\nTime: " +str(format(float(j-90)/30,'.2f'))+"sec  Frame: "+str(j), fontsize = fontsize+2)

        # set the data in the axesimage object
        ctr=0
        im[ctr].set_array(P.green_stack[j]); ctr+=1
        im[ctr].set_array(P.green_stack[j]); ctr+=1

        # return the artists set
        return im

    n_frames = len(P.green_stack)
    ani = animation.FuncAnimation(fig, updatefig, frames=range(n_frames), interval=100, blit=False, repeat=True)

    ani.save(os.path.split(os.path.split(P.sua_filename)[0])[0]+'/movies/'+os.path.split(P.sua_filename)[1][:-5]+"_epoch"+str(P.selected_epoch)+"_unit"+str(unit)+'.mp4', writer=writer, dpi=P.animation_resolution)
    
    if P.plotting:
        plt.show()


def plot_figure(sua_filename, sta_array, unit, spikes, all_spikes, color_scheme, P):

    colors = P.colors

    #*************************************************************
    #************************ DEFINE ROI *************************
    #*************************************************************
    path_dir, fname = os.path.split(sua_filename)
    Define_roi(sta_array, P)        #Use 92nd frame to draw brainmap;

    #print roi_coords[0], roi_coords[1]
    
    stmtd = []
    for j in range(len(P.coords)):
        stmtd.append([])
        for k in range(len(sta_array)):
            stmtd[j].append(sta_array[k][int(float(P.coords[j][0]))][int(float(P.coords[j][1]))])

    stmtd=np.array(stmtd)*100.

    #********************* PLOT ROI STMTD ***********************
    font_size = P.font_size
    fig = plt.figure()
    ax=plt.subplot(121)
    for k in range(len(stmtd)):
        plt.plot(stmtd[k], color=colors[k], linewidth=6)
        
    plt.plot([0, len(stmtd[0])], [0,0], 'r--', linewidth =3, color='black', alpha=0.8)
    plt.plot([int(len(stmtd[0])/2),int(len(stmtd[0])/2)], [-25,25], 'r--', linewidth =3, color='black', alpha=0.8)
    
    vmax_ylim = np.max(np.abs(stmtd))
    plt.ylim(-vmax_ylim-1, vmax_ylim+1)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.ylabel("DF/F (%)", fontsize=font_size, fontweight='bold')
    
    new_xlabel = np.arange(-3.0, 3.1, 1)
    old_xlabel = np.linspace(0, len(stmtd[0]), 7)
    plt.xticks(old_xlabel, new_xlabel, fontsize=font_size)
    plt.xlim(0, 180)
    plt.xlabel("Time (sec)", fontsize=font_size, fontweight='bold')
    
    #********PLOT LEGEND
    patches = []
    for k in range(len(P.coords)):
        patches.append(mpatches.Patch(color = P.colors[k]))
    
    labels = np.loadtxt(os.path.split(os.path.split(P.sua_filename)[0])[0]+"/roi_names.txt", dtype=str)

    legend = ax.legend(patches, labels, fontsize=15, loc=0, title="ROI Time Courses")
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=font_size-5)
    legend.get_title().set_fontsize(font_size-5)


    #**************************************************************
    #************************ PLOT IMG STACK **********************
    #**************************************************************

    block_save=P.block_save
    img_out = []
    start = int(P.start*30+90); end = int(P.end*30+90)
    for i in range(start, end, block_save):
        #img_out.append(np.ma.average(sta_array[i:i+block_save], axis=0))
        img_out.append(np.mean(sta_array[i:i+block_save], axis=0))
        
    img_out_nomask = np.array(img_out)

    midline_mask = P.midline_mask
    path_dir = os.path.split(os.path.split(sua_filename)[0])[0]
    img_out =  mask_data(img_out, P)
    
    
    v_max = vmax_ylim*1E-2; v_min = -v_max
    img_out = np.ma.hstack((img_out))
    
    #Make t=0 sec bar in time stack
    img_out[:,len(img_out[1])/2-3:len(img_out[1])/2]=v_min
    
    fig = plt.figure()
    ax=plt.subplot(111)

    im = plt.imshow(img_out, vmin=v_min, vmax=v_max, cmap=color_scheme)

    print img_out.shape
    new_xlabel = np.round(np.linspace(P.start,P.end, 7),2)
    old_xlabel = np.linspace(0, img_out.shape[1], 7)
    print new_xlabel
    print old_xlabel
    plt.xticks(old_xlabel, new_xlabel, fontsize=25)
    
    plt.xlabel("Time from event (sec)", fontsize = font_size, fontweight = 'bold')
    #ax.get_yaxis().set_visible(False)
    #ax.get_xaxis().set_visible(False)
    plt.yticks([])
    plt.ylabel("#"+str(unit+1), fontsize = font_size, fontweight = 'bold')
    
    #********************* PLOT COLOR BAR ******************
    cbar = fig.colorbar(im, ticks = [v_min, 0, v_max], ax=ax, fraction=0.02, pad=0.05, aspect=3)
    cbar.ax.set_yticklabels([str(round(v_min*100,1))+"%", '0'+"%", str(round(v_max*100,1))+"%"])  # vertically oriented colorbar
    cbar.ax.tick_params(labelsize=20) 
        
    
    plt.suptitle("Cell: "+str(unit) + " # spikes in epoch: "+ str(len(spikes))+" / "+str(len(all_spikes))+"\nDF/F max: "+str(round(v_max*100,1))+"%", fontsize=30)

    plt.show()

    plt.close(fig)
