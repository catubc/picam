import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import struct
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

   
colors = ['b','r', 'g', 'c','m','y','k','b','r']

filename = '/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/imaging/track_1_spontaneous_1.raw'

stream = open(filename, 'rb')

width = 128
height = 128
frames = 17979
Y = np.fromfile(stream, dtype=np.uint8, count=width*height*frames*3).reshape((frames, height, width, 3))
Y = np.float32(Y)


for k in range(len(Y)):
    Y[k] = np.flipud(np.fliplr(Y[k]) )

color_map = "jet" #"jet"

#***********GENERATE ANIMATIONS
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
