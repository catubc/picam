import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

colors = ['b','r', 'g', 'c','m','y','k','b','r']


filenames = [
'/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/imaging/track_1_whiskerstim_1.raw_time.txt',
'/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/imaging/track_1_whiskerstim_2.raw_time.txt',
'/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/imaging/track_2_whiskerstim_1.raw_time.txt',
#'/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/track_1_spontaneous_1_2.raw_time.txt',
'/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/imaging/track_1_whiskerstim_2_2.raw_time.txt',
'/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/imaging/track_1_spontaneous_1.raw_time.txt',
'/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/imaging/track_1_spontaneous_1_1.raw_time.txt',
'/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/imaging/track_1_spontaneous_1_3.raw_time.txt',
'/media/cat/12TB/in_vivo/tim/cat/2017_01_26_barrel_ephys_ophys/imaging/track_1_spontaneous_2.raw_time.txt'
]

for ctr, filename in enumerate(filenames):
    print ctr
    data = np.loadtxt(filename)*1E-6
    ax = plt.subplot(len(filenames),1,ctr+1)
    #plt.plot(data,color=colors[k])

    isi = data[1:]-data[:-1]
    bin_width = 0.01   # histogram bin width in usec
    y = np.histogram(isi, bins = np.arange(0, 2.0, bin_width))
    plt.bar(y[1][5:-1], y[0][5:], bin_width, color=colors[ctr])
    total_time = 0
    frame_ctr=0
    for k in range(5,len(y[0]),1):
        total_time += y[1][k]*y[0][k]
        frame_ctr += y[0][k]
        
    if ctr!=(len(filenames)-1):
        ax.get_xaxis().set_visible(False)
    else:
        plt.xlabel("Duration of Skipped Frame (sec)", fontsize=20)
        plt.tick_params(axis='x', which='both', labelsize=20)

    plt.ylim(0,10)
    plt.xlim(0,1)
    plt.ylabel("Rec len: "+str(round(np.max(data)/(60.),1))+" mins."+"\n # Frames skipped: "+str(frame_ctr)+"\n Skipped time: "+ \
            str(round(total_time ,3))+"sec.", rotation=0, labelpad=60, fontsize=10)

plt.suptitle("Picamera skipped frames analysis", fontsize=30)
plt.show()
