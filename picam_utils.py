#Support functions for picam code

import os
import numpy as np
import matplotlib.pyplot as plt
import parmap
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

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

def load_imaging_epochs(imaging_files, selected_epoch):
    
    root_dir = os.path.split(os.path.split(imaging_files[0])[0])[0]
    n_pixels = int(np.loadtxt(root_dir+'/n_pixels.txt'))

    #LOAD Imaging data for specific epoch
    imaging_epochs = []
    for ctr, filename in enumerate(imaging_files):
        if ctr!=selected_epoch: continue
        
        if os.path.exists(filename[:-4]+".npy")==False:
            stream = open(filename, 'rb')

            #Y = np.fromfile(stream, dtype=np.uint8, count=width*height*frames*3).reshape((frames, height, width, 3))
            Y = np.fromfile(stream, dtype=np.uint8).reshape((-1, n_pixels, n_pixels, 3))
        
            np.save(filename[:-4], Y)
            
            return Y[:,::-1,::-1,:]

        else:
            Y = np.load(filename[:-4]+'.npy', mmap_mode='c')
            return Y[:,::-1,::-1,:]

def set_blue_light(imaging_files, imaging_epoch, selected_epoch):
    
    imaging_onoff_file = imaging_files[selected_epoch][:-4]+'_imaging_onoff.txt'        #This sets the imaging_onoff for each recording in the track
    if os.path.exists(imaging_onoff_file)==False: 
        blue = imaging_epoch[:,64:65,50,1]
        lighton_trace = np.mean(blue, axis=1)
        plt.plot(lighton_trace)
        plt.show()
        print ">>>>> make file: ", imaging_onoff_file, "  <<<<<  and SAVE ONOFF TIMES"
    else:
        imaging_onoff = np.loadtxt(imaging_onoff_file, dtype=np.int32)


    imaging_epoch = imaging_epoch[imaging_onoff[0]:imaging_onoff[1]]

    return imaging_onoff, imaging_epoch


def set_frame_times(imaging_files, ephys_epochs, imaging_epochs, imaging_onoff, selected_epoch):
        
    frame_times = []
    imaging_file = imaging_files[selected_epoch]
    temp_times = np.loadtxt(imaging_file+"_time.txt",dtype=str)
    if temp_times[0]=="none":
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



def dff_mean(all_imaging, imaging_files, channel, selected_epoch):
    
    #************** MAKE GREEN CHANNEL
    imaging = all_imaging[:,:,:,channel]    #Flip UP-DOWN and LEFT-RIGHT while also selecting only Green channel

    stack_dff_filename = imaging_files[selected_epoch][:-4]+'_channel'+str(channel)+"_dffmean"
    if os.path.exists(stack_dff_filename+'.npy')==False:
        
        print "...computing mean..."
        baseline = np.mean(imaging, axis=0)
        print baseline.shape
        print baseline[64]
     
        n_processes = 20
        all_imaging_list = []
        list_indexes = np.int32(np.linspace(0, len(imaging), n_processes+1))
        print list_indexes
        for k in range(len(list_indexes[:-1])):                                             #**** DON"T NEED TO DO MAKE LIST FOR PARMAP ****
            all_imaging_list.append(imaging[list_indexes[k]:list_indexes[k+1]])
            
        print "...computing division..."

        dff_stack=[]
        dff_stack.extend(parmap.map(parallel_divide, all_imaging_list, baseline, processes = n_processes))
        print "...len parallel list: ", len(dff_stack)
        
        dff_stack = np.vstack(dff_stack)
        print dff_stack.shape
        
        np.save(stack_dff_filename, dff_stack)

    dff_stack = np.load(stack_dff_filename+'.npy', mmap_mode='c')
    print dff_stack.shape

    return dff_stack



def show_movies(dff_stack_green, dff_stack_blue):
    fig = plt.figure()
    ax1 = plt.subplot(231)
    plt.title("Green")

    ax2 = plt.subplot(232)
    plt.title("Blue")

    ax3 = plt.subplot(233)
    plt.title("Ratio")

    #ax4 = plt.subplot(234)
    #ax5 = plt.subplot(235)

    dff_max = 0.05
    #vmax = np.nanmax(np.abs(dff_stack_green)); print vmax
    im1 = ax1.imshow(dff_stack_green[0], vmin=-dff_max, vmax=dff_max)

    #vmax = np.nanmax(np.abs(dff_stack_blue)); print vmax
    im2 = ax2.imshow(dff_stack_blue[0], vmin=-dff_max, vmax=dff_max)

    #vmax = np.nanmax(np.abs(dff_stack_green))
    im3 = ax3.imshow(dff_stack_blue[0], vmin=0, vmax=10)
    #im4 = ax4.imshow(all_imaging[:,:,:,1][0], vmin=40, vmax=250)
    #im5 = ax5.imshow(all_imaging[:,:,:,2][0], vmin=40, vmax=250)


    def init():
        im1.set_data(np.zeros((128, 128)))
        im2.set_data(np.zeros((128, 128)))
        im3.set_data(np.zeros((128, 128)))
        #im4.set_data(np.zeros((128, 128)))
        #im5.set_data(np.zeros((128, 128)))
       

    def animate(i):
        k=i+0
        plt.suptitle("Frame: "+str(k))
        im1.set_data(dff_stack_green[k])
        im2.set_data(dff_stack_blue[k])
        im3.set_data(dff_stack_green[k]/dff_stack_blue[k])
        #im4.set_data(all_imaging[:,:,:,1][k])
        #im5.set_data(all_imaging[:,:,:,2][k])

        #return im1, im2, im3

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(dff_stack_green),
                                   interval=1)

    plt.show()     


               
def on_click_roi(event):
    
    global coords, images_temp, ax, fig, cid
    
    n_pix = len(images_temp[0])
    
    if event.inaxes is not None:
        coords.append((event.ydata, event.xdata))
        for j in range(len(coords)):
            for k in range(3):
                for l in range(3):
                    images_temp[0][min(n_pix,int(coords[j][0])-1+k)][min(n_pix,int(coords[j][1])-1+l)]=0

        ax.imshow(images_temp[0], vmin=-0.05, vmax=0.05)
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

        ax.imshow(images_processed, vmin=-0.05, vmax=0.05)#, vmin=0.0, vmax=0.02)
        ax.set_title("Select single pixel to track")
        cid = fig.canvas.mpl_connect('button_press_event', on_click_roi)
        plt.show()

        np.savetxt(genericmask_file, coords)
        
        return np.int16(coords)
    else:
        return np.int16(np.loadtxt(roi_file))




def mask_data(data, main_dir, midline_mask):
    
    n_pixels = len(data[0])
            
    #Load General mask (removes background)
    generic_mask_file = main_dir + '/genericmask.txt'        
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
    
    return temp_array














