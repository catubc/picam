import numpy as np

#import matplotlib
#matplotlib.use('Agg')  #Tkinter for some reason uses this.... otherwise get periodic crashes.

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import struct, os, sys

            
sys.path.append('/home/cat/code/')
import TSF.TSF as TSF
import PTCS.PTCS as PTCS
import dim_reduction.dim_reduction as dr

from picam_utils import *

class Parameters:
    pass

#**************************************************************************************

P = Parameters()

P.color_scheme = 'magma'     #viridis, plasma, inferno, magma
P.stack_color = 'green' #Name of channel to be processed
P.font_size = 25
P.animation_resolution = 100
P.smoothing_pixels = 2
P.midline_mask = 0

P.plotting = True
P.show_mask = True
P.show_movies = False
P.make_nopskip_arrays = False
P.dff_compute = True

P.block_save = 5
P.start = -1
P.end = +1
P.n_spikes = 25
P.imshow_scaling = 0.9

P.colors = ['b','r', 'g', 'c','m','y','k','b']
P.n_processes = 25

P.selected_epoch = 1
P.selected_unit = 0

print "... selected epoch: ", P.selected_epoch

#**************************************************************************************************************************
#*************************************************** GCAMP IMAGING ********************************************************
#**************************************************************************************************************************
#MOUSE BARREL - 2017_1_26 ------------ BEST EPOCHS: Epoch 1 has unit 1; Epoch 3 has unit 0; Epoch 4 has unit 2
#TSF FILE
#P.lfp_filename = '/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170126_153637_lfp_100hz_alltrack_50compressed_HIGHPASSED.tsf'
#LFP
#P.sua_filename = '/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170126_153637_lfp_100hz_alltrack_50compressed.ptcs'
#SUA
#sua_filename = '/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170126_153637_hp_butter_alltrack.ptcs'

#MOUSE VISUAL - 2017_02_03 ---------- EPOCH 0 WORKS (OTHERS MIGHT ALSO - NEED TO CHECK)
#TSF FILE
#P.lfp_filename = '/media/cat/12TB/in_vivo/tim/cat/2017_02_03_visual_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170203_172405_lfp_250hz_alltrack_50compressed_4.0threshold_3clusters_HIGHPASSED.tsf'
#LFP SORT - USE SPECIAL 3 CLUSTER FILE
#P.sua_filename = '/media/cat/12TB/in_vivo/tim/cat/2017_02_03_visual_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170203_172405_lfp_250hz_alltrack_50compressed_4.0threshold_3clusters.ptcs'

#MOUSE AUDITORY - 2017_01_30 - CLUSTERING NOT OPTIMAL, BUT WORKED BETTER WITH 50HZ LOWPASS
#TSF - MORE RECENT HIGHCUT ------------ EPOCHS 0 and 1
#P.lfp_filename = '/media/cat/12TB/in_vivo/tim/cat/2017_01_30_auditory_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170130_164612_lfp_250hz_alltrack_lowcut0.1_highcut110.0.tsf'
#LFP Sort
#P.sua_filename = '/media/cat/12TB/in_vivo/tim/cat/2017_01_30_auditory_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170130_164612_lfp_250hz_alltrack_lowcut0.1_highcut50_50compressed.ptcs'
#SUA
#sua_filename = '/media/cat/12TB/in_vivo/tim/cat/2017_01_30_auditory_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170130_164612_hp_butter_alltrack.ptcs'


#**************************************************************************************************************************
#*************************************************** VSD IMAGING **********************************************************
#**************************************************************************************************************************

#P.imaging_type = 'vsd'
#MOUSE BARREL - 2016_07_20      ---------- AUDITORY---------
#LFP
P.lfp_filename = '/media/cat/12TB/in_vivo/tim/cat/2016_07_20_vsd_barrel/sort_track2/track2_iso75_spontaneous_160720_220333_lfp_250hz_alltrack_lowcut0.1_highcut110.0.tsf'
#SUA
P.sua_filename = '/media/cat/12TB/in_vivo/tim/cat/2016_07_20_vsd_barrel/sort_track2/track2_iso75_spontaneous_160720_220333_lfp_250hz_alltrack_50compressed.ptcs'

#MOUSE VISUAL - 2016_7_15
#LFP
#sua_filename = '/media/cat/12TB/in_vivo/tim/cat/2016_07_15_vsd_visual/sort_alltrack_spontaneous/track1_150Hz_1st_spontaneous_10iso_160715_181445_lfp_250hz_alltrack_50compressed.ptcs'
#SUA
#sua_filename = '/media/cat/12TB/in_vivo/tim/cat/2016_07_15_vsd_visual/sort_alltrack_spontaneous/track1_150Hz_1st_spontaneous_10iso_160715_181445_hp_alltrack.ptcs'

#MOUSE AUDITORY - 2016_7_26
#LFP
#sua_filename = '/media/cat/12TB/in_vivo/tim/cat/2016_07_26_vsd_auditory/sort_alltrack2/track2_spontaneous_1iso_160726_215426_lfp_250hz_alltrack_50compressed.ptcs'
#SUA
#sua_filename = '/media/cat/12TB/in_vivo/tim/cat/2016_07_26_vsd_auditory/sort_alltrack2/track2_spontaneous_1iso_160726_215426_hp_butter_alltrack.ptcs'

#MOUSE AUDITORY - 2016_8_29 - ADDITIONAL 
#LFP
#sua_filename = '/media/cat/12TB/in_vivo/tim/cat/2016_08_29_vsd_auditory/sort_alltrack/track1_noisechirp_10repeats_10sec_1ms_160829_190523_lfp_250hz_alltrack_50compressed.ptcs'
#SUA
#sua_filename = '/media/cat/12TB/in_vivo/tim/cat/2016_08_29_vsd_auditory/sort_alltrack/track1_noisechirp_10repeats_10sec_1ms_160829_190523_hp_butter_alltrack.ptcs'


#*************************************************************************************************************************
#*************************************************************************************************************************
#*************************************************************************************************************************


if 'vsd' in P.sua_filename: 
    P.imaging_type = 'vsd'
else:
    P.imaging_type = 'gcamp'

P.main_dir = os.path.split(os.path.split(P.sua_filename)[0])[0]

if P.dff_compute: 

    #Load Single Unit data
    P.sort_sua =  PTCS.PTCS(P.sua_filename)
    print "... # units: ", len(P.sort_sua.units)
    for k in range(P.sort_sua.n_units):
        print "...unit: ", k, " # events: ", len(P.sort_sua.units[k])

    P.ephys_epochs = find_ephys_epochs(P)


    #*************************** LOAD LIST OF IMAGING FILES ******************
    P.alltrack_imaging_filenames = os.path.split(P.sua_filename)[0]+'/imaging_files.txt'
    P.imaging_files = np.loadtxt(P.alltrack_imaging_filenames, dtype=str)


    #****** LOAD IMAGING 
    load_imaging(P)     #Converts .raw files to .npy; saves raw imaging in: P.imaging_original 
    

    #****** FIND BLUE LIGHT
    #Set blue light visually from imaing files; then save to imaging_onoff.txt; clip imaging periods also
    set_lighton(P)


    #****** FILTER GREEN IMAGING STACK

    #Load raw imaging data and save to .npy files; mmap on subsequent loads
    filter_imaging(P)


    #****** LOAD FRAME TIMES AND ASIGN TO IMAGING FRAMES
    set_frame_times(P)

    print P.all_frame_times
    print P.imaging_filtered_lighton.shape

    #plt.imshow(all_imaging[10000])
    #plt.show()
    #quit()

    #**************************************************************************************************
    #******************************   CAN LOOP UP TO HERE TO LOAD MULTIPLE IMAGING SESSIONS   *********
    #**************************************************************************************************

    #*********************** COMPUTE DF/F USING GLOBAL SIGNAL REGRESSION

    dff_mean(P)


    ##********************** VERIFY MOVIES OF GREEN FLUORESCENCE ***********************

    #if False: 
        #show_dff_movies(dff_stack_green, sua_filename)


    #*********************** FIND SPIKE TRIGGERED INDEXES **************************
    if True: 
        for unit in range(P.sort_sua.n_units):
            
            if unit!=P.selected_unit: 
                continue
            
            #Make green fluorescence map
            compute_sta(unit, P)

            if  P.epoch_spikes < 25:
                print "... too few locking spikes for movies..."
                continue

            if P.show_movies: show_movies_2by1(unit, P)
        
        quit()

else:
    #******************************** DIM REDUCTION METHODS *************************

    files = [
    #MOUSE BARREL - GCAMP
    #'/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170126_153637_lfp_100hz_alltrack_50compressed_epoch3_unit0_colorgreen_sta_array_noskip.npy',
    #'/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170126_153637_lfp_100hz_alltrack_50compressed_epoch1_unit1_colorgreen_sta_array_noskip.npy',
    #'/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170126_153637_lfp_100hz_alltrack_50compressed_epoch4_unit2_colorgreen_sta_array_noskip.npy'

    #MOUSE VISUAL - GCAMP
    #'/media/cat/12TB/in_vivo/tim/cat/2017_02_03_visual_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170203_172405_lfp_250hz_alltrack_50compressed_4.0threshold_3clusters_epoch0_unit0_colorgreen_sta_array_noskip.npy',
    #'/media/cat/12TB/in_vivo/tim/cat/2017_02_03_visual_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170203_172405_lfp_250hz_alltrack_50compressed_4.0threshold_3clusters_epoch0_unit1_colorgreen_sta_array_noskip.npy',
    #'/media/cat/12TB/in_vivo/tim/cat/2017_02_03_visual_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170203_172405_lfp_250hz_alltrack_50compressed_4.0threshold_3clusters_epoch0_unit2_colorgreen_sta_array_noskip.npy'
    #SSD DATA
    #'/media/cat/500GB/track_1_spontaneous_1_170203_172405_lfp_250hz_alltrack_50compressed_4.0threshold_3clusters_epoch0_unit0_colorgreen_sta_array_noskip.npy',
    #'/media/cat/500GB/track_1_spontaneous_1_170203_172405_lfp_250hz_alltrack_50compressed_4.0threshold_3clusters_epoch0_unit1_colorgreen_sta_array_noskip.npy',
    #'/media/cat/500GB/track_1_spontaneous_1_170203_172405_lfp_250hz_alltrack_50compressed_4.0threshold_3clusters_epoch0_unit2_colorgreen_sta_array_noskip.npy'


    #MOUSE AUDITORY - GCAMP
    '/media/cat/12TB/in_vivo/tim/cat/2017_01_30_auditory_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170130_164612_lfp_250hz_alltrack_lowcut0.1_highcut50_50compressed_epoch0_unit0_colorgreen_sta_array_noskip.npy',
    '/media/cat/12TB/in_vivo/tim/cat/2017_01_30_auditory_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170130_164612_lfp_250hz_alltrack_lowcut0.1_highcut50_50compressed_epoch0_unit1_colorgreen_sta_array_noskip.npy',
    '/media/cat/12TB/in_vivo/tim/cat/2017_01_30_auditory_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170130_164612_lfp_250hz_alltrack_lowcut0.1_highcut50_50compressed_epoch1_unit2_colorgreen_sta_array_noskip.npy'
    #SSD
    #'/media/cat/500GB/track_1_spontaneous_1_170130_164612_lfp_250hz_alltrack_lowcut0.1_highcut50_50compressed_epoch0_unit0_colorgreen_sta_array_noskip.npy',
    #'/media/cat/500GB/track_1_spontaneous_1_170130_164612_lfp_250hz_alltrack_lowcut0.1_highcut50_50compressed_epoch0_unit1_colorgreen_sta_array_noskip.npy',
    #'/media/cat/500GB/track_1_spontaneous_1_170130_164612_lfp_250hz_alltrack_lowcut0.1_highcut50_50compressed_epoch1_unit2_colorgreen_sta_array_noskip.npy'
    ]

    print "... loading files ..."

    cluster_stacks_file = os.path.split(files[0])[0]+"/sta_array_clusters"

    if os.path.exists(cluster_stacks_file+'.npy')==False: 
        cluster_stacks = []
        for file_ in files:
            print "...loading: ", file_
            data = np.load(file_, mmap_mode = 'c')
            print data.shape
            P.n_pixels = data.shape[3]
            
            #UNRAVEL LAST DIMENSION OF DATA - Need this in order to compute mean below... not usre absolutely necessary...
            data_array = []
            for k in range(len(data)):
            #for k in range(100):
                data_array.append([])
                for j in range(len(data[k])):
                    temp_ravel = np.ravel(data[k][j])
                    data_array[k].append(temp_ravel)
            
            data_array = np.array(data_array)
            data_array = np.swapaxes(data_array, 0,1)
            print data_array.shape
            
            ave_stack = []
            for k in range(len(data_array)):
                ave_stack.append(np.mean(data_array[k], axis=0))
            
            ave_stack = np.array(ave_stack)
            print ave_stack.shape
            
            ave_stack_2d = ave_stack.reshape((180,P.n_pixels,P.n_pixels))   
            print ave_stack_2d.shape
            ave_stack = mask_data(ave_stack_2d, P)              #MASK off brain pixels for normalization computation below
            
            if True:                                            #NB: Must preserve real zero on normalization; ************************
                print "... normalizing masked stacks ..."
                #vmax = np.nanmax(ave_stack); vmin = np.nanmin(ave_stack)       #This doesn't preserve 0-centred data
                vmax_zero_centred = np.nanmax(np.abs(ave_stack))
                print "original max and min: ", vmax_zero_centred
                ave_stack = np.divide(ave_stack, vmax_zero_centred)
            
            print np.max(ave_stack), np.min(ave_stack)

            temp_array = []
            for k in range(len(ave_stack)):
                temp_array.append(np.ravel(ave_stack[k]))
            temp_array = np.array(temp_array)
            print temp_array.shape
            
            cluster_stacks.extend(temp_array)
            
            print "\n\n"
        cluster_stacks = np.array(cluster_stacks)
        print cluster_stacks.shape

        np.save(cluster_stacks_file, cluster_stacks)

    else:
        cluster_stacks = np.load(cluster_stacks_file+".npy")


    methods = ['MDS', 'tSNE', 'PCA', 'BH_tSNE']
    method = 2

    filename = P.sua_filename

    if False:
        traces_array = cluster_stacks
        min_index=180
    else:
        min_index=60
        traces_array=[]
        for k in range(3):
            traces_array.extend(cluster_stacks[k*180+60:k*180+120])

        traces_array = np.array(traces_array)

    recompute = True
    data_out = dr.dim_reduction_general(traces_array, method, filename, recompute)
    print data_out.shape


    temp_out = []
    for k in range(3):
        temp_chunk = data_out[k*min_index:(k+1)*min_index]
        print temp_chunk.shape
        temp_out.append(temp_chunk)

    data_out = np.array(temp_out)
    print data_out.shape
    #*************** CLUSTER DATA *********************

    #color_array = []
    #for k in range(112):
    #    for p in range(180):
    #        color_array.append(p)

    #******** PLOT OUTPUT DATA ***************
    import matplotlib.cm as cm

    if (method == 2) or (method == 0): 

        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        #ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=48, azim=134)
        ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=90, azim=90)

        
        #cmap = plt.cm.get_cmap('viridis', min_index)
        #plt.set_cmap('viridis')
        
        #for k in range(0, len(data_out[0])-1,1):
        #    #ax.scatter(data_out[0][k], data_out[1][k], data_out[2][k], s=100, c=cm.viridis(int(float(k%min_index)/min_index*256)))
        #    #ax.plot([data_out[0][k], data_out[0][k+1]], [data_out[1][k], data_out[1][k+1]],[data_out[2][k], data_out[2][k+1]], linewidth=3, c=cm.viridis(int(float(k%min_index)/min_index*256)))
        #    ax.scatter(data_out[0][k], data_out[1][k], data_out[2][k], s=100, c=P.colors[int(k/float(min_index))])
        #    ax.plot([data_out[0][k], data_out[0][k+1]], [data_out[1][k], data_out[1][k+1]],[data_out[2][k], data_out[2][k+1]], linewidth=3, c=P.colors[int(k/float(min_index))])
        #ax.scatter(data_out[0][k+1], data_out[1][k+1], data_out[2][k+1], s=100, c=P.colors[int(k/float(min_index))])

        for k in range(len(data_out[0])-1):
            ax.scatter(data_out[0][k][0], data_out[0][k][1], data_out[0][k][2], s=200, c=cm.Purples(int(float(k%min_index)/min_index*256)), alpha=0.75)
            ax.plot([data_out[0][k][0], data_out[0][k+1][0]], [data_out[0][k][1], data_out[0][k+1][1]],[data_out[0][k][2], data_out[0][k+1][2]], linewidth=6, c=cm.Purples(int(float(k%min_index)/min_index*256)), alpha=0.75)
        k=k+1
        ax.scatter(data_out[0][k][0], data_out[0][k][1], data_out[0][k][2], s=200, c=cm.Purples(int(float(k%min_index)/min_index*256)), alpha=0.75)
        
        
        for k in range(len(data_out[1])-1):
            ax.scatter(data_out[1][k][0], data_out[1][k][1], data_out[1][k][2], s=200, c=cm.Greys(int(float(k%min_index)/min_index*256)), alpha=0.75)
            ax.plot([data_out[1][k][0], data_out[1][k+1][0]], [data_out[1][k][1], data_out[1][k+1][1]],[data_out[1][k][2], data_out[1][k+1][2]], linewidth=6, c=cm.Greys(int(float(k%min_index)/min_index*256)), alpha=0.75)
        k=k+1
        ax.scatter(data_out[1][k][0], data_out[1][k][1], data_out[1][k][2], s=200, c=cm.Greys(int(float(k%min_index)/min_index*256)), alpha=0.75)

        for k in range(len(data_out[2])-1):
            ax.scatter(data_out[2][k][0], data_out[2][k][1], data_out[2][k][2], s=200, c=cm.Oranges(int(float(k%min_index)/min_index*256)), alpha=0.75)
            ax.plot([data_out[2][k][0], data_out[2][k+1][0]], [data_out[2][k][1], data_out[2][k+1][1]],[data_out[2][k][2], data_out[2][k+1][2]], linewidth=6, c=cm.Oranges(int(float(k%min_index)/min_index*256)), alpha=0.75)
        k=k+1
        ax.scatter(data_out[2][k][0], data_out[2][k][1], data_out[2][k][2], s=200, c=cm.Oranges(int(float(k%min_index)/min_index*256)), alpha=0.75)


        #ax.set_xlabel('PC1', fontsize=45, labelpad = 30)
        #ax.set_ylabel('PC2', fontsize=45, labelpad = 30)
        #ax.set_zlabel('PC3', fontsize=45, labelpad = 15)
        plt.title("PCA                ", fontsize = 60,  weight = 'bold')
        
    if method ==3: 
        for k in range(0, len(data_out[0])-1, 1):
            pass
            #plt.scatter(data_out[0][k], data_out[1][k], s=50, c=color_array[k], c=cm.viridis(int(float(k)/min_index*256)))
        
        plt.title("tSNE (BH)", fontsize = 60,  weight = 'bold')
        plt.xticks([])
        plt.yticks([])

    plt.show()

    #fig = plt.figure(2, figsize=(4, 3))
    #plt.clf()


quit()

































#***********GENERATE ANIMATIONS
color_map = "jet" #"jet"


Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=3000)

fig = plt.figure()
im = []

#gs = gridspec.GridSpec(2,len(self.ca_stack)*2)
gs = gridspec.GridSpec(2,3)

#[Ca] stacks
titles = ["Green", "Short Blue", "Ratio"]

#Green stack
baseline = np.mean(Y[1000:2000,:,:,1], axis=0)
green_stack = (Y[1000:2000,:,:,1]-baseline)/baseline

ax = plt.subplot(gs[0,0])
ax.get_xaxis().set_visible(False); ax.yaxis.set_ticks([]); ax.yaxis.labelpad = 0
#vmax = np.nanmax(np.abs(green_stack)); vmin=-vmax
vmax = 0.05; vmin=-vmax
plt.title("Green\n(DF/F: "+str(round(vmax*100,1))+"%)", fontsize = 12)
im.append(plt.imshow(green_stack[0], vmin=vmin, vmax=vmax,  cmap=color_map, interpolation='none'))
 

#Short blue stack
baseline = np.mean(Y[1000:2000,:,:,2], axis=0)
blue_stack = (Y[1000:2000,:,:,2]-baseline)/baseline

ax = plt.subplot(gs[0,1])
ax.get_xaxis().set_visible(False); ax.yaxis.set_ticks([]); ax.yaxis.labelpad = 0
#vmax = np.nanmax(np.abs(blue_stack)); vmin=-vmax
vmax = 0.05; vmin=-vmax
plt.title("Blue\n(DF/F: "+str(round(vmax*100,1))+"%)", fontsize = 12)
im.append(plt.imshow(blue_stack[0], vmin=vmin, vmax=vmax, cmap=color_map, interpolation='none'))


#Red
baseline = np.mean(Y[1000:2000,:,:,0], axis=0)
red_stack = (Y[1000:2000,:,:,0]-baseline)/baseline

ax = plt.subplot(gs[0,2])
ax.get_xaxis().set_visible(False); ax.yaxis.set_ticks([]); ax.yaxis.labelpad = 0
#vmax = np.nanmax(np.abs(blue_stack)); vmin=-vmax
vmax = 1.; vmin=-vmax
plt.title("Red\n(DF/F: "+str(round(vmax*100,1))+"%)", fontsize = 12)
im.append(plt.imshow(red_stack[0], vmin=vmin, vmax=vmax, cmap=color_map, interpolation='none'))


#Ratio stack
ratio_stack = np.divide(green_stack, blue_stack)

ax = plt.subplot(gs[1,0])
ax.get_xaxis().set_visible(False); ax.yaxis.set_ticks([]); ax.yaxis.labelpad = 0
#vmax = np.nanmax(np.abs(ratio_stack)); vmin=-vmax
vmax = 5; vmin=0
plt.title("Green / Blue", fontsize = 12)
im.append(plt.imshow(ratio_stack[0], vmin=vmin, vmax=vmax,cmap=color_map, interpolation='none'))

#Subtraction stack
subtraction_stack = green_stack - blue_stack

ax = plt.subplot(gs[1,1])
ax.get_xaxis().set_visible(False); ax.yaxis.set_ticks([]); ax.yaxis.labelpad = 0
#vmax = np.nanmax(np.abs(subtraction_stack)); vmin=-vmax
vmax = 0.05; vmin=-vmax
plt.title("Green - Blue\n(DF/F: "+str(round(vmax*100,1))+"%)", fontsize = 12)
im.append(plt.imshow(subtraction_stack[0], vmin=vmin, vmax=vmax, cmap=color_map,interpolation='none'))



#Loop to combine all video insets into 1
def updatefig(j):
    print "...frame: ", j
    #plt.suptitle(self.selected_dff_filter+'  ' +self.dff_method + "\nFrame: "+str(j)+"  " +str(format(float(j)/self.img_rate-self.parent.n_sec,'.2f'))+"sec  ", fontsize = 15)
    plt.suptitle("Time: " +str(format(float(j)/30,'.2f'))+"sec  Frame: "+str(j), fontsize = 15)

    # set the data in the axesimage object
    ctr=0
    im[ctr].set_array(green_stack[j]); ctr+=1
    im[ctr].set_array(blue_stack[j]); ctr+=1
    im[ctr].set_array(red_stack[j]); ctr+=1
    im[ctr].set_array(ratio_stack[j]); ctr+=1
    im[ctr].set_array(subtraction_stack[j]); ctr+=1

    # return the artists set
    return im

n_frames = 500
# kick off the animation
ani = animation.FuncAnimation(fig, updatefig, frames=range(n_frames), interval=100, blit=False, repeat=True)
#ani = animation.FuncAnimation(fig, updatefig, frames=range(len(self.ca_stack[1])), interval=100, blit=False, repeat=True)

if True:
    #ani.save(self.parent.root_dir+self.parent.animal.name+"/movie_files/"+self.selected_session+'_'+str(len(self.movie_stack))+'_'+str(self.selected_trial)+'trial.mp4', writer=writer, dpi=300)
    ani.save(filename[:-4]+"_"+color_map+'.mp4', writer=writer, dpi=600)
plt.show()


quit()
    
    
    

for k in range(1000,2000,10):
    
    plt.imshow(Y[k,:,:,1]-baseline)
    plt.show()

quit()



quit()

from bitstring import ConstBitStream
s = ConstBitStream(filename=filename)
data = s.readlist('883703808*intle:24')

print "...done reading..."
print "... converting to numpy..."
data = np.array(data).reshape(3,128,128,17979)
print data.shape()

np.save(filename, data)

quit()
bytes_read = open(filename, "rb").read()

data = struct.unpack('<I', 883703808)

print data
quit()



data = np.array(bytes_read).reshape(128,128,3,)
print data.shape()

quit()

for b in bytes_read:
    process_byte(b)












quit()
#CODE TO LOAD AND PARSE TRIGGERS FROM EPHYS EXTRA CHANNELS
filename = '/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/ephys/track_1_spontaneous_1_170126_153637_channel_0.npy'

data = np.int8(np.load(filename))

#np.save(filename, data)

print data.shape
print type(data[0])

down_indexes = np.where(data==0)[0]
up_indexes = np.where(data==1)[0]

time_indexes = []
time_indexes.append(up_indexes[0])
for k in range(1, len(up_indexes)-10, 10):
    if (up_indexes[k]+10)!=up_indexes[k+10]:
        time_indexes.append(up_indexes[k])
        time_indexes.append(up_indexes[k+10])

time_indexes.append(up_indexes[-1])
print time_indexes


#plt.plot(data)
#plt.show()
