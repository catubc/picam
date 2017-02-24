import numpy as np

#import matplotlib
#matplotlib.use('Agg')  #Tkinter for some reason uses this.... otherwise get periodic crashes.

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import struct, os, sys

#from tools import tsf_ptcs_classes as tsf_ptcs 

sys.path.append('/home/cat/code/')
import TSF.TSF as TSF
import PTCS.PTCS as PTCS

from picam_utils import *


#**************************************************************************************

colors = ['b','r', 'g', 'c','m','y','k','b','r']
n_processes = 10

selected_epoch = 1

print "... selected epoch: ", selected_epoch
#************************* LOAD SPIKING FOR SINGLE RECORDING OR WHOLE TRACK *****************************
#sua_filename = '/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170126_153637_hp_butter_alltrack.ptcs'
#sua_filename = '/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170126_153637_lfp_100hz_alltrack_50compressed.ptcs'

sua_filename = '/media/cat/12TB/in_vivo/tim/cat/2017_01_27_auditory_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170127_172223_lfp_250hz_alltrack_50compressed.ptcs'
#sua_filename = '/media/cat/12TB/in_vivo/tim/cat/2017_01_27_auditory_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170127_172223_hp_butter_alltrack.ptcs'

sort_sua =  PTCS.PTCS(sua_filename)
print "... # units: ", len(sort_sua.units)


#*********************** FIND BLUE LIGHT ON/OFF IN EPHYS *********************
lfp_filename = sua_filename.replace('hp_butter_alltrack.ptcs', 'lfp_250hz_alltrack.tsf')

ephys_epochs = find_ephys_epochs(lfp_filename, selected_epoch)


#*************************** LOAD LIST OF IMAGING FILES ******************
alltrack_imaging_filenames = os.path.split(sua_filename)[0]+'/imaging_files.txt'

imaging_files = np.loadtxt(alltrack_imaging_filenames, dtype=str)

#*************************** FIND BLUE LIGHT ON/OFF IN IMAGING ********************

#Load raw imaging data and save to .npy files; mmap on subsequent loads
imaging_epoch = load_imaging_epochs(imaging_files, selected_epoch)

#Set blue light visually from imaing files; then save to imaging_onoff.txt; clip imaging periods also
imaging_onoff, imaging_epochs = set_blue_light(imaging_files, imaging_epoch, selected_epoch)


#****************** LOAD FRAME TIMES AND ASIGN TO IMAGING FRAMES ***************
all_frametimes, all_imaging = set_frame_times(imaging_files, ephys_epochs, imaging_epochs, imaging_onoff, selected_epoch)

print all_frametimes
print all_imaging.shape

#**************************************************************************************************
#******************************   CAN LOOP UP TO HERE TO LOAD MULTIPLE IMAGING SESSIONS   *********
#**************************************************************************************************

#*********************** COMPUTE DF/F USING GLOBAL SIGNAL REGRESSION

channel = 1
dff_stack_green = dff_mean(all_imaging, imaging_files, channel, selected_epoch)

channel = 2
dff_stack_blue = dff_mean(all_imaging, imaging_files, channel, selected_epoch)


#********************** VERIFY MOVIES OF GREEN FLUORESCENCE ***********************

if False: 
    show_dff_movies(dff_stack_green, dff_stack_blue, sua_filename)


#*********************** FIND SPIKE TRIGGERED INDEXES **************************

units = np.arange(0,sort_sua.n_units,1)
#offset_frame_times = all_frametimes

for unit in units:
    plotting = False
    
    #Make green fluorescence map
    green_sta, n_spikes = make_sta_maps(unit, selected_epoch, sort_sua, all_frametimes, dff_stack_green, sua_filename, plotting, 'green')

    if n_spikes<100: 
        print "... too few locking spikes..."
        continue
        
    #Make blue light map
    blue_sta, n_spikes = make_sta_maps(unit, selected_epoch, sort_sua, all_frametimes, dff_stack_blue, sua_filename, plotting, 'blue')
    
    show_movies(unit, selected_epoch, green_sta, blue_sta, sua_filename, n_spikes)


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
