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

def find_ephys_epochs(lfp_filename, selected_epoch):
    epoch_file = os.path.split(lfp_filename)[0]+"/ephys_epochs.txt"
    if os.path.exists(epoch_file)==False: 

        tsf = TSF.TSF(lfp_filename)
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
        
    return ephys_epochs[selected_epoch]

def rotate_imaging(unit, sta_array, sua_filename):
    
    root_dir = os.path.split(os.path.split(sua_filename)[0])[0]
    n_pixels = int(np.loadtxt(root_dir+'/n_pixels.txt'))
    image_shift_file = sua_filename[:-5]+"_image_shift.txt"

    #rotated_imaging_file = sua_filename[:-5]+"_unit"+str(unit)+"_rotated.npy"

    sta_copy = sta_array.copy()

    #if os.path.exists(rotated_imaging_file)==False:
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
    
    sta_array = np.float64(sta_array)
    
    #ax = plt.subplot(221)
    #plt.imshow(sta_copy[90])

    #ax = plt.subplot(222)
    #plot_2 = np.roll(sta_copy[90], y_shift, axis=0)
    #plot_2[n_pixels/2] = 0
    #plot_2[:, n_pixels/2] = 0
    #plt.imshow(plot_2)
    
    #ax = plt.subplot(223)
    #plot_3 = np.roll(sta_copy[90], x_shift, axis=1)
    #plot_3 = np.roll(plot_3, y_shift, axis=0)
    #plot_3[n_pixels/2] = 0
    #plot_3[:, n_pixels/2] = 0
    #plt.imshow(plot_3)
    
    #ax = plt.subplot(224)
    #plot_4 = skimage.transform.rotate(sta_copy[90], angle)#, mode='constant', cval=100)
    #plot_4 = np.roll(plot_4, x_shift, axis=1)
    #plot_4 = np.roll(plot_4, y_shift, axis=0)
    #plot_4[n_pixels/2] = 0
    #plot_4[:, n_pixels/2] = 0
    #plt.imshow(plot_4)

    #plt.show()


    #*******************ROTATE IMAGE********************
    print "Rotate angle: ", angle
    if angle != 0:

        for i in range(len(sta_copy)):
            sta_copy[i] = skimage.transform.rotate(sta_copy[i], angle)#, mode='constant', cval=100)

            sta_copy[i] = np.roll(sta_copy[i], x_shift, axis=1)
            sta_copy[i] = np.roll(sta_copy[i], y_shift, axis=0)


    return sta_copy
    
    

def filter_imaging(imaging_files, selected_epoch, sua_filename, n_processes):
    
    print "...loading imaging epochs..."
    root_dir = os.path.split(os.path.split(imaging_files[0])[0])[0]
    n_pixels = int(np.loadtxt(root_dir+'/n_pixels.txt'))

    #LOAD Imaging data for specific epoch
    #for ctr, filename in enumerate(imaging_files):
    #    if ctr!=selected_epoch: continue
    filename = imaging_files[selected_epoch]
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
    
    
    #**************** COMPUTE MEAN OF STACK ************************
    
    if os.path.exists(filename[:-4]+"_epoch"+str(selected_epoch)+"_mean.npy")==False:
        print "... computing imaging mean..."
        Y_mean = np.mean(Y, axis=0)
        
        np.save(filename[:-4]+"_epoch"+str(selected_epoch)+"_mean", Y_mean)
    else:
        
        Y_mean = np.load(filename[:-4]+"_epoch"+str(selected_epoch)+"_mean.npy")
        
    
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
        filtered_pixels = parmap.map(do_filter, pixel_array, b, a, processes=n_processes)
        

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


    return Y_filtered[:,::-1,::-1]     #THIS INVERTS THE DATA; IT IS SAVED RIGHT SIDE UP THOUGH

           
def do_filter(pixel, b, a):
    return np.float16(scipy.signal.filtfilt(b, a, pixel))



def set_blue_light(imaging_files, imaging_epoch, selected_epoch):
    print "...setting blue light..."
  
    root_dir = os.path.split(os.path.split(imaging_files[selected_epoch])[0])[0]
    n_pixels = int(np.loadtxt(root_dir+'/n_pixels.txt'))
    imaging_onoff_file = imaging_files[selected_epoch][:-4]+'_imaging_onoff.txt'        #This sets the imaging_onoff for each recording in the track

    if os.path.exists(imaging_onoff_file)==False: 
        
        Y = np.load(imaging_files[selected_epoch][:-4]+'.npy', mmap_mode='c')
        
        imaging_epoch = Y[:,:,:,1]     
        
        #intensity = np.mean(imaging_epoch[:1000].reshape(1000,65536), axis=1)
        #plt.plot(intensity)
        #plt.show()

        blue = imaging_epoch[:,int(n_pixels/2):int(n_pixels/2)+1,128]
        
        lighton_trace = np.mean(blue, axis=1)

        print ">>>>> make file: \n\n", imaging_onoff_file, "\n\n  <<<<<  and SAVE ONOFF TIMES"

        plt.plot(lighton_trace)
        plt.show()

    else:
        imaging_onoff = np.loadtxt(imaging_onoff_file, dtype=np.int32)


    imaging_epoch = imaging_epoch[imaging_onoff[0]:imaging_onoff[1]]

    return imaging_onoff, imaging_epoch
    

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
    




def set_frame_times(imaging_files, ephys_epochs, imaging_epochs, imaging_onoff, selected_epoch):
    ''' Clip imaging stack to match on/off times of light
    '''
    
    
    print "...setting frame times..."

    frame_times = []
    imaging_file = imaging_files[selected_epoch]
    temp_times = np.loadtxt(imaging_file+"_time.txt",dtype=str)
    if temp_times[0]=="None":
        frame_times= np.int64(temp_times[1:])
    else:
        frame_times = np.int64(temp_times)
    
    print frame_times, len(frame_times)
    print imaging_onoff
    print ephys_epochs

    #***************** ASIGN REAL TIME TO IMAGING FRAMES ******************
    all_frametimes = []
    all_imaging = []

    print "* Rec epoch: ", selected_epoch
    print "... start ephys: ", ephys_epochs[0]/25000.
    print "... duration ephys: ", (ephys_epochs[1]-ephys_epochs[0])/25000.

    #Check if recording light was properly turned off
    if (imaging_onoff[1] - imaging_onoff[0])<=0:
        print " ****************** error in imaging ****************** \n"
        return

    temp_frametimes = frame_times[imaging_onoff[0]:imaging_onoff[1]]       #Clip frame times to on/off of light
    temp_frametimes = temp_frametimes - temp_frametimes[0]                          #Offset all imgframes to zero start
    temp_frametimes = temp_frametimes*1E-6                                          #Convert frametimes to seconds
    temp_frametimes = temp_frametimes+ephys_epochs[0]/25000.                     #Offset imgtimes to ephys trigger

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
        all_imaging_npy_file = imaging_files[0][:-4]+"_allimaging"

        if os.path.exists(all_imaging_npy_file+'.npy')==False:
            all_imaging = np.vstack(all_imaging)
            np.save(all_imaging_npy_file, all_imaging)
        else:
            all_imaging = np.load(all_imaging_npy_file+'.npy', mmap_mode='c')

    print all_imaging.shape
    print len(all_frametimes)
    print all_frametimes


    return all_frametimes, all_imaging



def dff_mean(imaging, imaging_files, selected_epoch, n_processes):
    
    print "...computing dff..."


    #************** MAKE GREEN CHANNEL
    #imaging = all_imaging[:,:,:,channel]    #Flip UP-DOWN and LEFT-RIGHT while also selecting only Green channel

    stack_dff_filename = imaging_files[selected_epoch][:-4]+"_green_dffmean"
    if os.path.exists(stack_dff_filename+'.npy')==False:
        
        print "...loading mean..."
        baseline = np.load(imaging_files[selected_epoch][:-4]+"_epoch"+str(selected_epoch)+"_mean.npy")
        
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
            
        dff_stack = np.zeros(imaging.shape, dtype=np.float32)
        for k in range(len(imaging)):
            print "...dividing frame: ", k
            #dff_stack[k]=np.divide(imaging[k]-baseline, baseline)
            dff_stack[k]=np.divide(imaging[k], baseline)
            
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

    return dff_stack



def show_dff_movies(dff_stack_green, sua_filename):

    
    midline_mask = 0
    main_dir = os.path.split(os.path.split(sua_filename)[0])[0]

    #temp_stack = dff_stack_green[50000:51000]
    temp_stack = dff_stack_green[:1000]
    temp_stack = (temp_stack - np.mean(temp_stack))/np.mean(temp_stack)

    dff_stack_green = mask_data(temp_stack, main_dir, midline_mask)
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
    
    global coords, images_temp, ax, fig, cid
    
    n_pix = len(images_temp)
    
    if event.inaxes is not None:
        coords.append((event.ydata, event.xdata))
        for j in range(len(coords)):
            for k in range(3):
                for l in range(3):
                    images_temp[0][min(n_pix,int(coords[j][0])-1+k)][min(n_pix,int(coords[j][1])-1+l)]=0

        ax.imshow(images_temp, vmin=-.25, vmax=.25)
        fig.canvas.draw()

    else:
        print 'Exiting'
        plt.close()
        fig.canvas.mpl_disconnect(cid)
        
def Define_roi(images_processed, main_dir):

    global coords, images_temp, ax, fig, cid
    
    images_temp = [images_processed.copy()]

    roi_file = main_dir + '/roi_coords.txt'

    if (os.path.exists(roi_file)==False):
        fig, ax = plt.subplots()
        coords=[]

        ax.imshow(images_processed, vmin=-.15, vmax=.15)#, vmin=0.0, vmax=0.02)
        ax.set_title("Select single pixel to track")
        cid = fig.canvas.mpl_connect('button_press_event', on_click_roi)
        plt.show()

        np.savetxt(roi_file, coords)
        
        return coords
    else:
        return np.loadtxt(roi_file, dtype=str)




def mask_data(data, main_dir, midline_mask):
    
    n_pixels = len(data[0])
            
    #Load General mask (removes background)
    generic_mask_file = main_dir + '/genericmask.txt' 
    if os.path.exists(generic_mask_file)==False:
        Define_generic_mask(np.array(data), main_dir)
           
    generic_coords = np.int32(np.loadtxt(generic_mask_file))
        
    generic_mask_indexes=np.zeros((n_pixels,n_pixels))
    for i in range(len(generic_coords)):
        generic_mask_indexes[generic_coords[i][0]][generic_coords[i][1]] = True

    #Load midline mask
    for i in range(midline_mask):
        generic_mask_indexes[:,n_pixels/2+int(midline_mask/2)-i]=True
        
    temp_array = np.ma.array(np.zeros((len(data),n_pixels,n_pixels),dtype=np.float32), mask=True)
    #Mask all frames; NB: PROBABLY FASTER METHOD
    for i in range(0, len(data),1):
        temp_array[i] = np.ma.masked_array(data[i], mask=generic_mask_indexes)
    
    
    plt.imshow(generic_mask_indexes)
    plt.show()
    
    return temp_array



      
def on_click(event):
    
    global coords, images_temp, ax, fig, cid
    
    n_pix = len(images_temp[0])
    
    if event.inaxes is not None:
        coords.append((event.ydata, event.xdata))
        for j in range(len(coords)):
            for k in range(3):
                for l in range(3):
                    images_temp[int(len(images_temp)/2)][min(n_pix,int(coords[j][0])-1+k)][min(n_pix,int(coords[j][1])-1+l)]=1

        ax.imshow(images_temp[int(len(images_temp)/2)], vmin=-0.15, vmax=0.15)
        #plt.show()
        fig.canvas.draw()
                    #figManager = plt.get_current_fig_manager()
                    #figManager.window.showMaximized()
    else:
        print 'Exiting'
        plt.close()
        fig.canvas.mpl_disconnect(cid)


       
def Define_generic_mask(images_processed, main_dir):

    global coords, images_temp, ax, fig, cid
    
    images_temp = images_processed.copy()
    
    print images_processed.shape

    if (os.path.exists(main_dir + '/genericmask.txt')==False):
        fig, ax = plt.subplots()
        coords=[]

        print images_temp[int(len(images_temp)/2)]
        ax.imshow(images_temp[int(len(images_temp)/2)], vmin=-0.15, vmax=0.15)
        ax.set_title("Compute generic (outside the brain) mask")
        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

        #******* MASK AND DISPLAY AREAS OUTSIDE GENERAL MASK 
        #Search points outside and black them out:
        all_points = []
        for i in range(len(images_processed[0][0])):
            for j in range(len(images_processed[0][0])):
                all_points.append([i,j])

        all_points = np.array(all_points)
        vertixes = np.array(coords) 
        vertixes_path = Path(vertixes)
        
        mask = vertixes_path.contains_points(all_points)
        counter=0
        coords_save=[]
        for i in range(len(images_processed[0][0])):
            for j in range(len(images_processed[0][0])):
                if mask[counter] == False:
                    images_processed[int(len(images_temp)/2)][i][j]=0
                    coords_save.append([i,j])
                counter+=1

        fig, ax = plt.subplots()
        ax.imshow(images_processed[int(len(images_temp)/2)])
        plt.show()

        np.savetxt(main_dir+'/genericmask.txt', coords_save)


def make_sta_maps(unit, selected_epoch, sort_sua, all_frame_times, dff_stack, sua_filename, plotting, stack_color):

#for unit in units:
    print "... cell: ", unit
    
    sta_map_indexes = []
    for k in range(180):
        sta_map_indexes.append([])

    if "50compressed" in sua_filename: 
        all_spikes = np.float32(sort_sua.units[unit])*1E-6*50.
    else: 
        all_spikes = np.float32(sort_sua.units[unit])*1E-6 #/25000.
    print "... no. spikes: ", len(all_spikes)
    
    if len(all_spikes)==0: 
        print "... cell has no spikes..."
        return [], 0
    
    frame_stack = []
    dt = 0.0333         #Timewindow to search for nearest frame
    #for spike in spikes[:1000]: #use only first 1000 spikes
    #    nearest_frames = find_nearest_180(offset_frame_times, spike, dt)
    #    frame_stack.append(nearest_frames)
    
    n_processes = 25
    
    
    #************************* EXCLUDE SPIKES OUTSIDE OF IMAGING PERIOD ********************

    spike_indexes = np.where(np.logical_and(all_spikes>=all_frame_times[0], all_spikes<=all_frame_times[-1]))[0]    #Exclude spikes too close to beginning or end of recordings.
    spikes = all_spikes[spike_indexes]
    
    #**************** FIND NEAREST FRAME TIMES FOR EACH SPIKE **************
    
    frame_stack_file = sua_filename[:-5]+"_epoch"+str(selected_epoch)+"_unit"+str(unit)+"_frame_stack"
    if os.path.exists(frame_stack_file+".npy")==False: 
        print "... finding frame indexes for # spikes: ", len(spikes), ' / ', len(all_spikes)

        frame_stack.append(parmap.map(find_nearest_180_parallel, spikes, all_frame_times, dt, processes = n_processes))

        print "... done..."
        frame_stack = np.vstack(frame_stack)
        print frame_stack.shape
        
        frame_stack = frame_stack.T
        np.save(frame_stack_file, frame_stack)
    else:
        frame_stack = np.load(frame_stack_file+'.npy')

    #********************** COMPUTE STA MAPS **********************

    sta_array_file = sua_filename[:-5]+"_epoch"+str(selected_epoch)+"_unit"+str(unit)+"_color"+stack_color+"_sta_array"
    if os.path.exists(sta_array_file+".npy")==False:
    
        #sta_array.append(parmap.map(make_stack_parallel, frame_stack, dff_stack_green, processes = n_processes))

        sta_array=[]
        print "...generating frames..."
        for ctr,frames in enumerate(frame_stack):               #****************** PARALLELIZE THIS!!!!
            #ax = plt.subplot(10,18,ctr+1)                      #************ ALSO REMOVE REDUNDANT SPIKES THAT ARE OUTSIDE OF WINDOW
            frames_temp = []
            for k in frames:
                if k != None: 
                    frames_temp.append(k)
            print "...frame: ", ctr, "  # of indexes: ", len(frames_temp)
            
            temp_ave = np.mean(dff_stack[np.int32(frames_temp)],axis=0)         #********** ALSO CAN PARALLELIZE THIS

            sta_array.append(temp_ave)
        
        sta_array = np.array(sta_array)
        np.save(sta_array_file, sta_array)
        
    else:
        sta_array = np.load(sta_array_file+'.npy')
    
    print sta_array.shape
    if len(sta_array)==0: 
        print "... cell has no spikes in epoch..."
        return [], 0


    #*************************** ROTATE IMAGING STACK ********************

    #Load raw imaging data and save to .npy files; mmap on subsequent loads
    sta_array_rotated = rotate_imaging(unit, sta_array, sua_filename)


    #********************* PLOTTING ***********************
    if plotting: 
        plot_figure(sua_filename, sta_array_rotated, unit, spikes, all_spikes)
    
    return sta_array_rotated, len(spikes)  #

def show_movies_2by1(unit, selected_epoch, green_stack, sua_filename, n_spikes):
    
    midline_mask = 5
    
    
    #*********** SMOOTHING
    if True: 
        for k in range(len(green_stack)):
            green_stack[k] = scipy.ndimage.filters.gaussian_filter(green_stack[k],2)
    
        
    #************ MASK DATA
    green_stack = mask_data(green_stack, os.path.split(os.path.split(sua_filename)[0])[0], midline_mask)
    #blue_stack = mask_data(blue_stack, os.path.split(os.path.split(sua_filename)[0])[0], midline_mask)
    
    
    #***********GENERATE ANIMATIONS
    color_map = "viridis" #"jet"

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
    vmax = np.nanmax(np.abs(green_stack)); vmin=-vmax
    print "...vmax: ", vmax
    #vmax = 0.05; vmin=-vmax
    plt.title("Green ("+str(round(vmin*100,1))+"-"+str(round(vmax*100,1))+"%)", fontsize = fontsize)
    im.append(plt.imshow(green_stack[0], vmin=vmin, vmax=vmax,  cmap=color_map, interpolation='none'))
    plt.ylabel("0% centred")
    
    #Green stack - Max dynamics
    ax = plt.subplot(gs[0,1])
    ax.get_xaxis().set_visible(False); ax.yaxis.set_ticks([]); ax.yaxis.labelpad = 0
    vmax = np.nanmax(np.abs(green_stack))*0.5; vmin=-vmax #np.nanmin(green_stack)
    #vmax = 0.05; vmin=-vmax
    plt.title("Green ("+str(round(vmin*100,1))+"-"+str(round(vmax*100,1))+"%)", fontsize = fontsize)
    im.append(plt.imshow(green_stack[0], vmin=vmin, vmax=vmax,  cmap=color_map, interpolation='none'))
    plt.ylabel("'Majid' plots")


    #Loop to combine all video insets into 1
    def updatefig(j):
        print "...animating frame: ", j, ' / ', n_frames
        #plt.suptitle(self.selected_dff_filter+'  ' +self.dff_method + "\nFrame: "+str(j)+"  " +str(format(float(j)/self.img_rate-self.parent.n_sec,'.2f'))+"sec  ", fontsize = 15)
        plt.suptitle("Unit: "+str(unit)+"  # spikes: "+ str(n_spikes)+"\nTime: " +str(format(float(j-90)/30,'.2f'))+"sec  Frame: "+str(j), fontsize = fontsize+2)

        # set the data in the axesimage object
        ctr=0
        im[ctr].set_array(green_stack[j]); ctr+=1
        im[ctr].set_array(green_stack[j]); ctr+=1

        # return the artists set
        return im

    n_frames = len(green_stack)
    ani = animation.FuncAnimation(fig, updatefig, frames=range(n_frames), interval=100, blit=False, repeat=True)

    ani.save(os.path.split(os.path.split(sua_filename)[0])[0]+'/movies/'+os.path.split(sua_filename)[1][:-5]+"_epoch"+str(selected_epoch)+"_unit"+str(unit)+'.mp4', writer=writer, dpi=100)
    
    plt.show()

def show_movies_2by2(unit, selected_epoch, green_stack, blue_stack, sua_filename, n_spikes):
    
    midline_mask = 0
    green_stack = mask_data(green_stack, os.path.split(os.path.split(sua_filename)[0])[0], midline_mask)
    blue_stack = mask_data(blue_stack, os.path.split(os.path.split(sua_filename)[0])[0], midline_mask)
    
    #***********GENERATE ANIMATIONS
    color_map = "viridis" #"jet"

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=3000)

    fig = plt.figure()
    im = []

    #gs = gridspec.GridSpec(2,len(self.ca_stack)*2)
    gs = gridspec.GridSpec(2,2)

    fontsize = 8

    #[Ca] stacks
    titles = ["Green", "Short Blue", "Ratio"]

    #Green stack
    ax = plt.subplot(gs[0,0])
    ax.get_xaxis().set_visible(False); ax.yaxis.set_ticks([]); ax.yaxis.labelpad = 0
    vmax = np.nanmax(np.abs(green_stack)); vmin=-vmax
    #vmax = 0.05; vmin=-vmax
    plt.title("Green ("+str(round(vmin*100,1))+"-"+str(round(vmax*100,1))+"%)", fontsize = fontsize)
    im.append(plt.imshow(green_stack[0], vmin=vmin, vmax=vmax,  cmap=color_map, interpolation='none'))
    plt.ylabel("0% centred")
    
    #Green stack - Max dynamics
    ax = plt.subplot(gs[1,0])
    ax.get_xaxis().set_visible(False); ax.yaxis.set_ticks([]); ax.yaxis.labelpad = 0
    vmax = np.nanmax(green_stack)*0.75; vmin=0 #np.nanmin(green_stack)
    #vmax = 0.05; vmin=-vmax
    plt.title("Green ("+str(round(vmin*100,1))+"-"+str(round(vmax*100,1))+"%)", fontsize = fontsize)
    im.append(plt.imshow(green_stack[0], vmin=vmin, vmax=vmax,  cmap=color_map, interpolation='none'))
    plt.ylabel("'Majid' plots")


    #Short blue stack
    ax = plt.subplot(gs[0,1])
    ax.get_xaxis().set_visible(False); ax.yaxis.set_ticks([]); ax.yaxis.labelpad = 0
    vmax = np.nanmax(np.abs(blue_stack))*0.5; vmin=-vmax
    #vmax = 0.05; vmin=-vmax
    plt.title("Blue ("+str(round(vmin*100,1))+"-"+str(round(vmax*100,1))+"%)", fontsize = fontsize)
    im.append(plt.imshow(blue_stack[0], vmin=vmin, vmax=vmax, cmap=color_map, interpolation='none'))

    #Short blue stack - Max dynamics
    ax = plt.subplot(gs[1,1])
    ax.get_xaxis().set_visible(False); ax.yaxis.set_ticks([]); ax.yaxis.labelpad = 0
    vmax = np.nanmax(blue_stack)*0.25; vmin=0 #np.nanmin(blue_stack)
    #vmax = 0.05; vmin=-vmax
    plt.title("Blue ("+str(round(vmin*100,1))+"-"+str(round(vmax*100,1))+"%)", fontsize = fontsize)
    im.append(plt.imshow(blue_stack[0], vmin=vmin, vmax=vmax, cmap=color_map, interpolation='none'))



    ##Ratio stack
    #ratio_stack = np.divide(green_stack, blue_stack)
    #ax = plt.subplot(gs[0,2])
    #ax.get_xaxis().set_visible(False); ax.yaxis.set_ticks([]); ax.yaxis.labelpad = 0
    ##vmax = np.nanmax(np.abs(ratio_stack)); vmin=-vmax
    ##vmax = 20; vmin=-10

    #data_1d = ratio_stack.ravel()

    #vmax = np.percentile(data_1d, 0.80)               #Mark stroke as the 97.5 percentile and higher values; 
    #vmin = np.percentile(data_1d, 0.2)               #Mark stroke as the 97.5 percentile and higher values; 

    #print "...ratio vmax: ", vmax, vmin
    #plt.title("Green / Blue", fontsize = 12)
    #im.append(plt.imshow(ratio_stack[0], vmin=vmin, vmax=vmax,cmap=color_map, interpolation='none'))

    #Loop to combine all video insets into 1
    def updatefig(j):
        print "...animating frame: ", j, ' / ', n_frames
        #plt.suptitle(self.selected_dff_filter+'  ' +self.dff_method + "\nFrame: "+str(j)+"  " +str(format(float(j)/self.img_rate-self.parent.n_sec,'.2f'))+"sec  ", fontsize = 15)
        plt.suptitle("Unit: "+str(unit)+"  # spikes: "+ str(n_spikes)+"\nTime: " +str(format(float(j)/30,'.2f'))+"sec  Frame: "+str(j), fontsize = fontsize+2)

        # set the data in the axesimage object
        ctr=0
        im[ctr].set_array(green_stack[j]); ctr+=1
        im[ctr].set_array(green_stack[j]); ctr+=1
        im[ctr].set_array(blue_stack[j]); ctr+=1
        im[ctr].set_array(blue_stack[j]); ctr+=1
        #im[ctr].set_array(ratio_stack[j]); ctr+=1
        #im[ctr].set_array(subtraction_stack[j]); ctr+=1

        # return the artists set
        return im

    n_frames = len(green_stack)
    # kick off the animation
    ani = animation.FuncAnimation(fig, updatefig, frames=range(n_frames), interval=100, blit=False, repeat=True)
    #ani = animation.FuncAnimation(fig, updatefig, frames=range(len(self.ca_stack[1])), interval=100, blit=False, repeat=True)

    print os.path.split(os.path.split(sua_filename)[0])[0]+'/movies/'+os.path.split(sua_filename)[1][:-5]+"_epoch"+str(selected_epoch)+"_unit"+str(unit)+'.mp4'
    
    if True:
        #ani.save(self.parent.root_dir+self.parent.animal.name+"/movie_files/"+self.selected_session+'_'+str(len(self.movie_stack))+'_'+str(self.selected_trial)+'trial.mp4', writer=writer, dpi=300)
        ani.save(os.path.split(os.path.split(sua_filename)[0])[0]+'/movies/'+os.path.split(sua_filename)[1][:-5]+"_epoch"+str(selected_epoch)+"_unit"+str(unit)+'.mp4', writer=writer, dpi=600)
    plt.show()



def plot_figure(sua_filename, sta_array, unit, spikes, all_spikes):

    colors = ['b','r', 'g', 'c','m','y','k','b','r']

    #************************ DEFINE ROI TO TRACK ***********************
    path_dir, fname = os.path.split(sua_filename)
    roi_coords = Define_roi(sta_array[92], path_dir)        #Use 92nd frame to draw brainmap;
    #roi_coords = Define_roi(np.mean(sta_array), path_dir)        #Use 92nd frame to draw brainmap;
    #print roi_coords.split()

    print roi_coords[0], roi_coords[1]
    
    stmtd = []
    for j in range(len(roi_coords)):
        stmtd.append([])
        for k in range(len(sta_array)):
            stmtd[j].append(sta_array[k][int(float(roi_coords[j][0]))][int(float(roi_coords[j][1]))])
    

    stmtd=np.array(stmtd)*100.
    
    #******************** MASK AND AVERAGE DATA ***************
    block_save=5
    img_out = []
    start = 60; length = 60
    for i in range(start, start+length, block_save):
        img_out.append(np.ma.average(sta_array[i:i+block_save], axis=0))
        
    midline_mask = 10
    
    path_dir = os.path.split(os.path.split(sua_filename)[0])[0]
    img_out =  mask_data(img_out, path_dir, midline_mask)
    
    v_max = np.nanmax(np.abs(img_out))*0.75; v_min = -v_max
    img_out = np.ma.hstack((img_out))
    
    #Make midline bar
    img_out[:,len(img_out[1])/2-3:len(img_out[1])/2]=v_min
    
    
    fig = plt.figure()
    ax=plt.subplot(211)

    im = plt.imshow(img_out, vmin=v_min, vmax=v_max, cmap="viridis")

    #new_xlabel = np.arange(-3.0+start*0.033, -3.0+(start+length)*0.033, 0.5)
    #old_xlabel = np.linspace(0, img_out.shape[1], 7)
    #plt.xticks(old_xlabel, new_xlabel, fontsize=25)
    
    plt.xlabel("Time from spike (sec)", fontsize = 30)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    
    #********************* PLOT COLOR BAR ******************
    cbar = fig.colorbar(im, ticks = [v_min, 0, v_max], ax=ax, fraction=0.02, pad=0.05, aspect=3)
    cbar.ax.set_yticklabels([str(round(v_min*100,1))+"%", '0'+"%", str(round(v_max*100,1))+"%"])  # vertically oriented colorbar
    cbar.ax.tick_params(labelsize=15) 
        
    
    #********************* PLOT STMTD ***********************
    ax=plt.subplot(212)
    for k in range(len(stmtd)):
        plt.plot(stmtd[k], color=colors[k], linewidth=3)
        
    plt.plot([0, len(stmtd[0])], [0,0], 'r--', color='black')
    plt.plot([int(len(stmtd[0])/2),int(len(stmtd[0])/2)], [-25,25], 'r--', color='black')
    plt.ylim(np.min(stmtd), np.max(stmtd))
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.ylabel("DF/F (%)", fontsize=25)
    
    new_xlabel = np.arange(-3.0, 3.1, 1)
    old_xlabel = np.linspace(0, len(stmtd[0]), 7)
    plt.xticks(old_xlabel, new_xlabel, fontsize=25)
    plt.xlim(0, 180)
    plt.xlabel("Time (sec)", fontsize=25)
    
    plt.suptitle("Cell: "+str(unit) + " # spikes in epoch: "+ str(len(spikes))+" / "+str(len(all_spikes))+"\nDF/F max: "+str(round(v_max*100,1))+"%", fontsize=30)
    plt.show()
    
    plt.close(fig)
