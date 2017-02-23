import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import struct, os, sys
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from tools import tsf_ptcs_classes as tsf_ptcs 

sys.path.append('/home/cat/code/')
import TSF.TSF as TSF
import PTCS.PTCS as PTCS

colors = ['b','r', 'g', 'c','m','y','k','b','r']


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
        
#************************* LOAD SPIKING *****************************

#ephys_filename= '/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/ephys/track_1_spontaneous_1_170126_153637_hp.ptcs'
sua_filename = '/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170126_153637_hp_butter_alltrack.ptcs'

sort_sua =  tsf_ptcs.Ptcs(sua_filename)

print "... # units: ", len(sort_sua.units)


#*********************** FIND BLUE LIGHT ON/OFF IN EPHYS *********************

lfp_filename = '/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/sort_alltrack_spontaneous/track_1_spontaneous_1_170126_153637_lfp_100hz_alltrack.tsf'

epoch_file = lfp_filename[:-4]+"_epochs.txt"
if os.path.exists(epoch_file)==False: 

    tsf = TSF.TSF(lfp_filename)
    tsf.read_footer()

    on_times = []
    off_times = []
    offset = 0

    for f in range(tsf.n_files[0]):
        #for k in range(tsf.n_digital_chs[f]):
        k = 0 #Only first channel contains on/off lights for imaging camera
        plt.plot(tsf.digital_chs[f][k][::10])

        #Find transitions to on and off
        for s in range(0, len(tsf.digital_chs[f][k])-25, 25):
            if tsf.digital_chs[f][k][s]!=tsf.digital_chs[f][k][s+25]:
                if tsf.digital_chs[f][k][s]==0: on_times.append(s+offset)
                else: off_times.append(s+offset)
        plt.show()
        offset = offset + tsf.n_samples[f] #Must offset future times by recording time;
        
    epochs = []
    for k in range(len(on_times)):
        epochs.append([on_times[k], off_times[k]])
    print epochs

    np.savetxt(epoch_file, epochs, fmt='%i')

else:
    epochs = np.loadtxt(epoch_file)


#*************************** LOAD LIST OF IMAGING FILES ******************

imaging_filenames = os.path.split(sua_filename)[0]+'/imaging_files.txt'
imaging_files = np.loadtxt(imaging_filenames, dtype=str)
print imaging_files
print imaging_files[0]


#*************************** FIND BLUE LIGHT ON/OFF IN IMAGING ********************
root_dir = os.path.split(os.path.split(sua_filename)[0])[0]

n_pixels = int(np.loadtxt(root_dir+'/n_pixels.txt'))

#LOAD Imaging data
imaging_epochs = []
for filename in imaging_files:

    if os.path.exists(filename[:-4]+".npy")==False:
        stream = open(filename, 'rb')

        #Y = np.fromfile(stream, dtype=np.uint8, count=width*height*frames*3).reshape((frames, height, width, 3))
        Y = np.fromfile(stream, dtype=np.uint8).reshape((-1, n_pixels, n_pixels, 3))
    
        np.save(filename[:-4], Y)
        
        imaging_epochs.append(Y)
    else:
        imaging_epochs.append(np.load(filename[:-4]+'.npy', mmap_mode='c'))


#Search imaging data for light on/off ************* FOR NOW MUST MANUALLY SET THESE ***************

imaging_onoff_file = os.path.split(sua_filename)[0]+'/imaging_onoff.txt'
if os.path.exists(imaging_onoff_file)==False: 
    for p in range(len(imaging_epochs)):
        blue = imaging_epochs[p][:,64:65,50,1]
        print blue.shape

        lighton_trace = np.mean(blue, axis=1)

else:
    imaging_onoff = np.loadtxt(imaging_onoff_file, dtype=np.int32)

print imaging_onoff
#Clip all the imaging records to the on/off periods

for e in range(len(imaging_epochs)):
    print imaging_epochs[e].shape
    imaging_epochs[e] = imaging_epochs[e][imaging_onoff[e][0]:imaging_onoff[e][1]]
    
    print imaging_epochs[e].shape, '\n'

#Y = np.float32(Y)
#on_off = [186,17217]
#Y = Y[on_off[0]:on_off[1]]
#for k in range(len(Y)):
#    Y[k] = np.flipud(np.fliplr(Y[k]) )
    
#quit()


#****************** LOAD FRAME TIMES ***************
for imaging_file in imaging_files:
    
    frame_times = np.loadtxt(imaging_file+"_time.txt",dtype=str)
    if frame_times[0]=="none":
        pass
    else:
        frame_times = np.int64(frame_times)
    print frame_times






quit()






epoch_file = '/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/ephys/track_1_spontaneous_1_170126_153637_epochs.txt'

epochs = np.loadtxt(epoch_file)

print epochs

imaging_start_offset = epochs[0][0]



#************************ MASK IMAGING DATA ***********************
path_dir, filename = os.path.split(filename)
tsf_ptcs.Define_generic_mask(Y, path_dir)



#************************ LOAD FRAME TIMES *************************

filename = '/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/imaging/track_1_spontaneous_1.raw_time.txt'
frame_times = np.loadtxt(filename)*1E-6

print frame_times
print "... # frames: ", len(frame_times)

offset_frame_times = frame_times[on_off[0]:on_off[1]]

offset_frame_times = offset_frame_times -offset_frame_times[0]

offset_frame_times = offset_frame_times+ imaging_start_offset

print offset_frame_times

#*********************** COMPUTE DF/F USING GLOBAL SIGNAL REGRESSION

#Select only single channel
data = Y[:,:,:,1]

if False:
    data = np.divide(Y[:,:,:,1], Y[:,:,:,2])


baseline = np.mean(data, axis=0)
dff_stack_green = (data-baseline)/baseline
    

print "...dff stack shape: ", dff_stack_green.shape


#*********************** FIND SPIKE TRIGGERED INDEXES **************************

units = np.arange(48,73,1)

for unit in units:
    print "... cell: ", unit
    sta_map_indexes = []
    for k in range(180):
        sta_map_indexes.append([])

    frame_stack = []
    spikes = np.float32(sort_sua.units[unit])/25000.
    indexes = np.where(np.logical_and(spikes>=offset_frame_times[0], spikes<=offset_frame_times[-1]))[0]
    spikes = spikes[indexes]
    
    if len(spikes)==0: 
        print "... no spikes..."
        continue
    
    print "... finding frame indexes for # spikes: ", len(spikes)
    for spike in spikes[:1000]: #use only first 1000 spikes
        nearest_frames = find_nearest_180(offset_frame_times, spike, 0.0333)
        #print nearest_frames
        frame_stack.append(nearest_frames)
        #print "... offset_frame_time: ", offset_frame_times[nearest_frame], "  spiketime: ", spike, "  found index: ", nearest_frame
        
        #quit()
        
    print "... done..."
    frame_stack = np.vstack(frame_stack)
    print frame_stack.shape
    print frame_stack[0]
    
    frame_stack = frame_stack.T

    #********************** COMPUTE STA MAPS **********************

    sta_array=[]
    print "...plotting frames..."
    for ctr,frames in enumerate(frame_stack):
        #ax = plt.subplot(10,18,ctr+1)
        frames_temp = []
        for k in frames:
            if k != None: 
                frames_temp.append(k)
        print "...frame: ", ctr, "  # of indexes: ", len(frames_temp)
        temp_ave = np.mean(dff_stack_green[np.int32(frames_temp)],axis=0)
        sta_array.append(temp_ave)
    
    
    #************************ DEFINE ROI TO TRACK ***********************
    path_dir, fname = os.path.split(filename)
    roi_coords = tsf_ptcs.Define_roi(sta_array[92], path_dir)
    print roi_coords[0], roi_coords[1]
    
    stmtd = []
    for k in range(len(sta_array)):
        stmtd.append(sta_array[k][roi_coords[0]][roi_coords[1]])

    stmtd=np.array(stmtd)*100.
    
    fig = plt.figure()

    ax=plt.subplot(211)
    block_save=10
    img_out = []
    start = 0; length = 179
    for i in range(start, start+length, block_save):
        img_out.append(np.ma.average(sta_array[i:i+block_save], axis=0))
    
    midline_mask = 0
    
    path_dir, fname = os.path.split(filename)
    img_out =  tsf_ptcs.mask_data(img_out, path_dir, midline_mask)
    
    v_max = np.nanmax(np.abs(img_out)); v_min = -v_max
    img_out = np.ma.hstack((img_out))
    
    #Make midline bar
    img_out[:,len(img_out[1])/2-3:len(img_out[1])/2]=v_min
    
    im = plt.imshow(img_out, vmin=v_min, vmax=v_max, cmap="jet")

    #new_xlabel = np.arange(-3.0+start*0.033, -3.0+(start+length)*0.033, 0.5)
    #old_xlabel = np.linspace(0, img_out.shape[1], 7)
    #plt.xticks(old_xlabel, new_xlabel, fontsize=25)
    
    plt.xlabel("Time from spike (sec)", fontsize = 30)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    
    #Plot color bar
    cbar = fig.colorbar(im, ticks = [v_min, 0, v_max], ax=ax, fraction=0.02, pad=0.05, aspect=3)
    cbar.ax.set_yticklabels([str(round(v_min*100,1))+"%", '0'+"%", str(round(v_max*100,1))+"%"])  # vertically oriented colorbar
    cbar.ax.tick_params(labelsize=15) 
        
    
    ax=plt.subplot(212)
    plt.plot(stmtd)
    plt.plot([0, len(stmtd)], [0,0], 'r--')
    plt.plot([int(len(stmtd)/2),int(len(stmtd)/2)], [-5,5], 'r--')
    plt.ylim(np.min(stmtd), np.max(stmtd))
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.ylabel("DF/F (%)", fontsize=25)
    
    new_xlabel = np.arange(-3.0, 3.1, 1)
    old_xlabel = np.linspace(0, len(stmtd), 7)
    plt.xticks(old_xlabel, new_xlabel, fontsize=25)
    plt.xlim(0, 180)
    plt.xlabel("Time (sec)", fontsize=25)
    
    plt.suptitle("Cell: "+str(unit) + " # spikes: "+str(len(spikes))+"\nDF/F max: "+str(round(v_max*100,1))+"%", fontsize=30)
    plt.show()
        


#for k in range(len(Y)):
#    Y[k] = np.flipud(np.fliplr(Y[k]) )




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
