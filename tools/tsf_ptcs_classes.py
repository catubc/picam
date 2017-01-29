import struct, array, csv
import numpy as np
import os
import math, operator
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pylab import *
import time
import itertools 
import h5py
import matplotlib.gridspec as gridspec
from matplotlib.path import Path


from pylab import get_current_fig_manager as gcfm
from scipy.signal import butter, lfilter, filtfilt
import scipy.io
from plotly.graph_objs import *
import plotly.plotly as py

colors = ['blue','red', 'black']

class Tsf_file(object):

    def __init__(self, fin, sim_dir):

        self.read_tsf(fin,sim_dir)

    def read_tsf(self,fin,sim_dir):

        self.header = fin.read(16)
        self.iformat = struct.unpack('i',fin.read(4))[0] 
        self.SampleFrequency = struct.unpack('i',fin.read(4))[0] 
        self.n_electrodes = struct.unpack('i',fin.read(4))[0] 
        self.n_vd_samples = struct.unpack('i',fin.read(4))[0] 
        self.vscale_HP = struct.unpack('f',fin.read(4))[0] 

        if self.iformat==1001:
            self.Siteloc = np.zeros((2*56), dtype=np.int16)
            self.Siteloc = struct.unpack(str(2*56)+'h', fin.read(2*56*2))
        if self.iformat==1002:
            self.Siteloc = np.zeros((2*self.n_electrodes), dtype=np.int16)
            self.Readloc = np.zeros((self.n_electrodes), dtype=np.int32)
            for i in range(self.n_electrodes):
                self.Siteloc[i*2] = struct.unpack('h', fin.read(2))[0]
                self.Siteloc[i*2+1] = struct.unpack('h', fin.read(2))[0]
                self.Readloc[i] = struct.unpack('i', fin.read(4))[0]

        print self.n_electrodes, self.n_vd_samples

        self.ec_traces =  np.fromfile(fin, dtype=np.int16, count=self.n_electrodes*self.n_vd_samples)
       
        self.ec_traces.shape = self.n_electrodes, self.n_vd_samples
        
        #f = open(sim_dir+'ulf.dat' , 'wb')
        #print "SAVING ULF / RAW data"
        #for i in range(self.n_electrodes):
            #self.ec_traces[i].tofile(f)

        self.n_cell_spikes = struct.unpack('i',fin.read(4))[0] 
        print "No. ground truth cell spikes: ", self.n_cell_spikes
        if (self.n_cell_spikes>0):
            if (self.iformat==1001):
                self.vertical_site_spacing = struct.unpack('i',fin.read(4))[0] 
                self.n_cell_spikes = struct.unpack('i',fin.read(4))[0] 

            self.fake_spike_times =  np.fromfile(fin, dtype=np.int32, count=self.n_cell_spikes)
            self.fake_spike_assignment =  np.fromfile(fin, dtype=np.int32, count=self.n_cell_spikes)
            self.fake_spike_channels =  np.fromfile(fin, dtype=np.int32, count=self.n_cell_spikes)

        if False and self.n_cell_spikes>0:  #Load ground truth data; NOT REALLY USED THIS WAY NOW; COULD BE IN FUTURE!
            #print self.fake_spike_assignment
            self.n_cells=max(self.fake_spike_assignment)
            print "No. of ground truth cell: ",self.n_cells
            self.cell_spikes=[[] for x in range(self.n_cells)]  #make lists of cells; .tsf files don't store #cells; could just read file twice;
            self.cell_ptp=[[] for x in range(self.n_cells)] 
            self.cell_size=[[] for x in range(self.n_cells)] 
            self.cell_maxchan=[[] for x in range(self.n_cells)] 
            self.cell_ptpsd=[[] for x in range(self.n_cells)] 

            for k in range(len(self.fake_spike_times)):
                self.cell_spikes[self.fake_spike_assignment[k]-1].append(self.fake_spike_times[k])
           
            ##Compute PTP for ground truth data
            if (os.path.exists(sim_dir+'/cell_ptp.csv')==False):
                for i in range(self.n_cells):
                    print "Computing PTP cell: ", i
                    self.compute_ptp(self.cell_spikes[i], self.n_electrodes, self.ec_traces, self.SampleFrequency, self.vscale_HP)
                    self.cell_ptp[i]=self.ptp
                    self.cell_size[i]=len(self.cell_spikes[i])
                    self.cell_maxchan[i]=self.maxchan
                    self.cell_ptpsd[i]=self.ptpsd
                    print "Cell: ", i , " ptp: ", self.cell_ptp[i], "ptp_sd: ", self.cell_ptpsd[i], " size: ", self.cell_size[i], " max ch: ", self.cell_maxchan[i]

                np.savetxt(sim_dir+'/cell_ptp.csv', self.cell_ptp, delimiter=",")
                np.savetxt(sim_dir+'/cell_ptpsd.csv', self.cell_ptpsd, delimiter=",")

    def compute_ptp(self, unit, n_electrodes, ec_traces, SampleFrequency, vscale_HP):


        ptp=[[0] for x in range(n_electrodes)]
        #CAN PLOT THE UNITS BY UNCOMMENTING SOME OF THESE STATEMENTS
        #x = np.arange(0,60,1)
        for i in range(n_electrodes):  
            #counter=0
            for k in unit:
                #print k
                if k>10: ptp[i] += vscale_HP*(max(ec_traces[i][k-10:k+10])-min(ec_traces[i][k-10:k+10]))
                
                #if (counter<50): plt.plot(x+int(i/15)*100, ec_traces[i][k-30:k+30]+(i%15)*100, color='black')
                #counter+=1

        #mng = plt.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        #plt.show()

        self.ptp = max(ptp)/float(len(unit))
        self.maxchan = np.argmax(ptp)
        
        #Compute SD of ptp on maxchan:
        ptp=np.zeros(len(unit), dtype=np.float32) 
        tempint=0
        for i in [self.maxchan-1]:
            for k in unit:
                if k>10:
                    ptp[tempint] = vscale_HP*(max(ec_traces[i][k-10:k+10])-min(ec_traces[i][k-10:k+10]))            
                    tempint+=1

        self.ptpsd = np.std(ptp)


class Loadptcs(object):
    """Polytrode clustered spikes file neuron record"""
    def __init__(self, sorted_file, work_dir, ptcs_flag, save_timestamps):
        
        f = open(work_dir+sorted_file+'.ptcs', "rb")
        self.sorted_file = sorted_file
        self.name = sorted_file
        self.full_path = work_dir + sorted_file
        # call the appropriate method:
        self.VER2FUNC = {1: self.readHeader, 2: self.readHeader, 3: self.readHeader}

        self.readHeader(f, ptcs_flag, save_timestamps)
        
        self.nid = []  #Make unique unit id list for loading later.
        
        self.loadData(self.nsamplebytes, f, work_dir, ptcs_flag, save_timestamps)
        
        f.close()

    def __getstate__(self):
        """Instance methods must be excluded when pickling"""
        d = self.__dict__.copy()
        try: del d['VER2FUNC']
        except KeyError: pass
        return d

    def readHeader(self, f, ptcs_flag, save_timestamps):
        """Read in neuron record of .ptcs file version 3. 'zpos' field was replaced
        by 'sigma' field.
        nid: int64 (signed neuron id, could be -ve, could be non-contiguous with previous)
        ndescrbytes: uint64 (nbytes, keep as multiple of 8 for nice alignment, defaults to 0)
        descr: ndescrbytes of ASCII text
        (padded with null bytes if needed for 8 byte alignment)
        clusterscore: float64
        xpos: float64 (um)
        ypos: float64 (um)
        sigma: float64 (um) (Gaussian spatial sigma)
        nchans: uint64 (num chans in template waveforms)
        chanids: nchans * uint64 (0 based IDs of channels in template waveforms)
        maxchanid: uint64 (0 based ID of max channel in template waveforms)
        nt: uint64 (num timepoints per template waveform channel)
        nwavedatabytes: uint64 (nbytes, keep as multiple of 8 for nice alignment)
        wavedata: nwavedatabytes of nsamplebytes sized floats
        (template waveform data, laid out as nchans * nt, in uV,
        padded with null bytes if needed for 8 byte alignment)
        nwavestdbytes: uint64 (nbytes, keep as multiple of 8 for nice alignment)
        wavestd: nwavestdbytes of nsamplebytes sized floats
        (template waveform standard deviation, laid out as nchans * nt, in uV,
        padded with null bytes if needed for 8 byte alignment)
        nspikes: uint64 (number of spikes in this neuron)
        spike timestamps: nspikes * uint64 (us, should be sorted)
        """

        self.nid = int(np.fromfile(f, dtype=np.int64, count=1)) # nid
        self.ndescrbytes = int(np.fromfile(f, dtype=np.uint64, count=1)) # ndescrbytes
        self.descr = f.read(self.ndescrbytes).rstrip('\0 ') # descr

        if self.descr:
            try:
                self.descr = eval(self.descr) # might be a dict
            except: pass

        self.nneurons = int(np.fromfile(f, dtype=np.uint64, count=1)) # nneurons
        self.nspikes = int(np.fromfile(f, dtype=np.uint64, count=1)) # nspikes
        self.nsamplebytes = int(np.fromfile(f, dtype=np.uint64, count=1)) # nsamplebytes
        self.samplerate = int(np.fromfile(f, dtype=np.uint64, count=1)) # samplerate
        self.npttypebytes = int(np.fromfile(f, dtype=np.uint64, count=1)) # npttypebytes

        self.pttype = f.read(self.npttypebytes).rstrip('\0 ') # pttype

        self.nptchans = int(np.fromfile(f, dtype=np.uint64, count=1)) # nptchans
        self.chanpos = np.fromfile(f, dtype=np.float64, count=self.nptchans*2) # chanpos
        self.chanpos.shape = self.nptchans, 2 # reshape into rows of (x, y) coords
        self.nsrcfnamebytes = int(np.fromfile(f, dtype=np.uint64, count=1)) # nsrcfnamebytes
        self.srcfname = f.read(self.nsrcfnamebytes).rstrip('\0 ') # srcfname
        # maybe convert this to a proper Python datetime object in the Neuron:
        self.datetime = float(np.fromfile(f, dtype=np.float64, count=1)) # datetime (days)
        self.ndatetimestrbytes = int(np.fromfile(f, dtype=np.uint64, count=1)) # ndatetimestrbytes
        self.datetimestr = f.read(self.ndatetimestrbytes).rstrip('\0 ') # datetimestr


    def loadData(self, n_bytes, f, work_dir, ptcs_flag, save_timestamps):
        #call the appropriate method:
        #self.VER2FUNC = {1: self.read_ver_1, 2:self.read_ver_2, 3:self.read_ver_3}
        self.nsamplebytes = n_bytes
        self.wavedtype = {2: np.float16, 4: np.float32, 8: np.float64}[self.nsamplebytes]

        self.n_units=self.nneurons
        #print self.nneurons
        self.units=[None]*self.n_units
        self.uid = [None]*self.n_units  #Unique id for full track sorts
        self.n_sorted_spikes = [None]*self.n_units
        self.ptp=np.zeros((self.n_units), dtype=np.float32)
        self.size = []
        self.maxchan = []
        
        #print "No. of units sorted: ", self.nneurons

        #print work_dir
        for k in range(self.n_units):
            self.readUnit(f,work_dir, ptcs_flag)
            self.units[k]= self.spikes
            #print "Unit: ", k, " spikes: ", self.spikes
            if 'martin' in self.full_path:
                self.uid[k]= self.nid
            else: #All other sorts are from Nick's SS so should be the same
                self.uid[k]= self.nid-1
               
            #print "SAMPLERATE: ", self.samplerate
            if ptcs_flag: #Martin's data has wrong flag for saves
                self.units[k]=[x*self.samplerate/1E+6 for x in self.units[k]] #Converts spiketimes from usec to timesteps
            else:
                self.units[k]=[x*self.samplerate/2/1E+6 for x in self.units[k]] #Converts spiketimes from usec to timesteps

            self.n_sorted_spikes[k] = len(self.units[k])
            self.size.append(self.nspikes)
            self.maxchan.append(self.maxchanu)
            self.ptp[k]=max(self.wavedata[np.where(self.chans==self.maxchanu)[0][0]]) - \
                        min(self.wavedata[np.where(self.chans==self.maxchanu)[0][0]]) #compute PTP of template;
            #self.ptp[k]=1
            #print self.name, " unit: ", k, " ptp: ", self.ptp[k], " maxchan: ", self.maxchanu

        f.close()

        ##Save .csv spike-times files for CK
        if save_timestamps: 
            for i in range(len(self.units)):
                with open(work_dir+"timestamps_"+str(i)+".csv", "w") as f:
                    writer = csv.writer(f)
                    for j in range(len(self.units[i])):
                        writer.writerow([round(float(self.units[i][j])/float(self.samplerate)*20,6)]) #LFP events X 20

        #if (os.path.exists(work_dir+self.sorted_file+'_ptps.csv')==False):
        #    np.savetxt(work_dir+self.sorted_file+'_ptps.csv', self.ptp, delimiter=",")
        #    np.savetxt(work_dir+self.sorted_file+'_size.csv', self.size, delimiter=",")

    def readUnit(self,f, work_dir, ptcs_flag):
        self.nid = int(np.fromfile(f, dtype=np.int64, count=1)) # nid
        self.ndescrbytes = int(np.fromfile(f, dtype=np.uint64, count=1)) # ndescrbytes
        self.descr = f.read(self.ndescrbytes).rstrip('\0 ') # descr

        if self.descr:
            try:
                self.descr = eval(self.descr) # might be a dict
            except: pass

        self.clusterscore = float(np.fromfile(f, dtype=np.float64, count=1)) # clusterscore
        self.xpos = float(np.fromfile(f, dtype=np.float64, count=1)) # xpos (um)
        self.ypos = float(np.fromfile(f, dtype=np.float64, count=1)) # ypos (um)
        self.zpos = float(np.fromfile(f, dtype=np.float64, count=1)) # zpos (um)
        self.nchans = int(np.fromfile(f, dtype=np.uint64, count=1)) # nchans
        self.chans = np.fromfile(f, dtype=np.uint64, count=self.nchans) #NB: Some errors here from older .ptcs formats
        self.maxchanu = int(np.fromfile(f, dtype=np.uint64, count=1)) # maxchanid

        self.nt = int(np.fromfile(f, dtype=np.uint64, count=1)) # nt: number of time points in template

        self.nwavedatabytes, self.wavedata = self.read_wave(f) #TEMPLATE

        self.nwavestdbytes, self.wavestd = self.read_wave(f) #STANDARD DEVIATION
        self.nspikes = int(np.fromfile(f, dtype=np.uint64, count=1)) # nspikes

        # spike timestamps (us):
        self.spikes = np.fromfile(f, dtype=np.uint64, count=self.nspikes)
        #print self.spikes
        #time.sleep(1)
        #print len(self.spikes)
        #quit()

        # convert from unsigned to signed int for calculating intervals:
        self.spikes = np.asarray(self.spikes, dtype=np.float64)

        #if 'nick' in work_dir:     #Use +1 indexes for channel #s
        if ptcs_flag: #0 = martin's sort
            #print "loading nick's data"
            #self.maxchanu-=1 #OLDER DATA
            pass
        else:
            pass
            #print "loading martin's data" #martin's data doesn't need to be converted
            #self.spikes=self.spikes #convert Martin's spike times to timesteps 
            #print self.spikes[-1]
            #quit()
            #print self.spikes
            #quit()
            
    def read_wave(self, f):
        """Read wavedata/wavestd bytes"""
        # nwavedata/nwavestd bytes, padded:
        nbytes = int(np.fromfile(f, dtype=np.uint64, count=1))
        fp = f.tell()
        count = nbytes // self.nsamplebytes # trunc to ignore any pad bytes
        X = np.fromfile(f, dtype=self.wavedtype, count=count) # wavedata/wavestd (uV)
        if nbytes != 0:
            X.shape = self.nchans, self.nt # reshape
        f.seek(fp + nbytes) # skip any pad bytes
        return nbytes, X

    def rstrip(s, strip):
        """What I think str.rstrip should really do"""
        if s.endswith(strip):
            return s[:-len(strip)] # strip it
        else:
            return s

    def read(self):
        self.nid = self.parse_id()
        with open(self.fname, 'rb') as f:
            self.spikes = np.fromfile(f, dtype=np.int64) # spike timestamps (us)
        self.nspikes = len(self.spikes)

    def read_tsf(self,f):
        pass


class Ptcs(object):
    """Polytrode clustered spikes file neuron record"""
    def __init__(self, sorted_file):
        
        f = open(sorted_file, "rb")
        self.sorted_file = sorted_file
        self.name = sorted_file
        self.full_path = sorted_file
        # call the appropriate method:
        self.VER2FUNC = {1: self.readHeader, 2: self.readHeader, 3: self.readHeader}

        self.readHeader(f)
        
        self.nid = []  #Make unique unit id list for loading later.
        
        self.loadData(self.nsamplebytes, f)
        
        f.close()

    def __getstate__(self):
        """Instance methods must be excluded when pickling"""
        d = self.__dict__.copy()
        try: del d['VER2FUNC']
        except KeyError: pass
        return d

    def readHeader(self, f):
        """Read in neuron record of .ptcs file version 3. 'zpos' field was replaced
        by 'sigma' field.
        nid: int64 (signed neuron id, could be -ve, could be non-contiguous with previous)
        ndescrbytes: uint64 (nbytes, keep as multiple of 8 for nice alignment, defaults to 0)
        descr: ndescrbytes of ASCII text
        (padded with null bytes if needed for 8 byte alignment)
        clusterscore: float64
        xpos: float64 (um)
        ypos: float64 (um)
        sigma: float64 (um) (Gaussian spatial sigma)
        nchans: uint64 (num chans in template waveforms)
        chanids: nchans * uint64 (0 based IDs of channels in template waveforms)
        maxchanid: uint64 (0 based ID of max channel in template waveforms)
        nt: uint64 (num timepoints per template waveform channel)
        nwavedatabytes: uint64 (nbytes, keep as multiple of 8 for nice alignment)
        wavedata: nwavedatabytes of nsamplebytes sized floats
        (template waveform data, laid out as nchans * nt, in uV,
        padded with null bytes if needed for 8 byte alignment)
        nwavestdbytes: uint64 (nbytes, keep as multiple of 8 for nice alignment)
        wavestd: nwavestdbytes of nsamplebytes sized floats
        (template waveform standard deviation, laid out as nchans * nt, in uV,
        padded with null bytes if needed for 8 byte alignment)
        nspikes: uint64 (number of spikes in this neuron)
        spike timestamps: nspikes * uint64 (us, should be sorted)
        """

        self.nid = int(np.fromfile(f, dtype=np.int64, count=1)) # nid
        self.ndescrbytes = int(np.fromfile(f, dtype=np.uint64, count=1)) # ndescrbytes
        self.descr = f.read(self.ndescrbytes).rstrip('\0 ') # descr

        if self.descr:
            try:
                self.descr = eval(self.descr) # might be a dict
            except: pass

        self.nneurons = int(np.fromfile(f, dtype=np.uint64, count=1)) # nneurons
        self.nspikes = int(np.fromfile(f, dtype=np.uint64, count=1)) # nspikes
        self.nsamplebytes = int(np.fromfile(f, dtype=np.uint64, count=1)) # nsamplebytes
        self.samplerate = int(np.fromfile(f, dtype=np.uint64, count=1)) # samplerate
        self.npttypebytes = int(np.fromfile(f, dtype=np.uint64, count=1)) # npttypebytes

        self.pttype = f.read(self.npttypebytes).rstrip('\0 ') # pttype

        self.nptchans = int(np.fromfile(f, dtype=np.uint64, count=1)) # nptchans
        self.chanpos = np.fromfile(f, dtype=np.float64, count=self.nptchans*2) # chanpos
        self.chanpos.shape = self.nptchans, 2 # reshape into rows of (x, y) coords
        self.nsrcfnamebytes = int(np.fromfile(f, dtype=np.uint64, count=1)) # nsrcfnamebytes
        self.srcfname = f.read(self.nsrcfnamebytes).rstrip('\0 ') # srcfname
        # maybe convert this to a proper Python datetime object in the Neuron:
        self.datetime = float(np.fromfile(f, dtype=np.float64, count=1)) # datetime (days)
        self.ndatetimestrbytes = int(np.fromfile(f, dtype=np.uint64, count=1)) # ndatetimestrbytes
        self.datetimestr = f.read(self.ndatetimestrbytes).rstrip('\0 ') # datetimestr


    def loadData(self, n_bytes, f):
        #call the appropriate method:
        #self.VER2FUNC = {1: self.read_ver_1, 2:self.read_ver_2, 3:self.read_ver_3}
        self.nsamplebytes = n_bytes
        self.wavedtype = {2: np.float16, 4: np.float32, 8: np.float64}[self.nsamplebytes]

        self.n_units=self.nneurons
        #print self.nneurons
        self.units=[None]*self.n_units
        self.uid = [None]*self.n_units  #Unique id for full track sorts
        self.n_sorted_spikes = [None]*self.n_units
        self.ptp=np.zeros((self.n_units), dtype=np.float32)
        self.size = []
        self.maxchan = []
        
        #print "No. of units sorted: ", self.nneurons

        #print work_dir
        for k in range(self.n_units):
            self.readUnit(f)
            self.units[k]= self.spikes
            #print "Unit: ", k, " spikes: ", self.spikes
            if 'martin' in self.full_path:
                self.uid[k]= self.nid
            else: #All other sorts are from Nick's SS so should be the same
                self.uid[k]= self.nid-1
               
            #print "SAMPLERATE: ", self.samplerate
            if 'martin' in self.full_path:
                self.units[k]=[x*self.samplerate/2/1E+6 for x in self.units[k]] #Converts spiketimes from usec to timesteps
            else:
                self.units[k]=[x*self.samplerate/1E+6 for x in self.units[k]] #Converts spiketimes from usec to timesteps

            self.n_sorted_spikes[k] = len(self.units[k])
            self.size.append(self.nspikes)
            self.maxchan.append(self.maxchanu)
            self.ptp[k]=max(self.wavedata[np.where(self.chans==self.maxchanu)[0][0]]) - \
                        min(self.wavedata[np.where(self.chans==self.maxchanu)[0][0]]) #compute PTP of template;
            #self.ptp[k]=1
            #print self.name, " unit: ", k, " ptp: ", self.ptp[k], " maxchan: ", self.maxchanu

        f.close()

       
    def readUnit(self,f):
        self.nid = int(np.fromfile(f, dtype=np.int64, count=1)) # nid
        self.ndescrbytes = int(np.fromfile(f, dtype=np.uint64, count=1)) # ndescrbytes
        self.descr = f.read(self.ndescrbytes).rstrip('\0 ') # descr

        if self.descr:
            try:
                self.descr = eval(self.descr) # might be a dict
            except: pass

        self.clusterscore = float(np.fromfile(f, dtype=np.float64, count=1)) # clusterscore
        self.xpos = float(np.fromfile(f, dtype=np.float64, count=1)) # xpos (um)
        self.ypos = float(np.fromfile(f, dtype=np.float64, count=1)) # ypos (um)
        self.zpos = float(np.fromfile(f, dtype=np.float64, count=1)) # zpos (um)
        self.nchans = int(np.fromfile(f, dtype=np.uint64, count=1)) # nchans
        self.chans = np.fromfile(f, dtype=np.uint64, count=self.nchans) #NB: Some errors here from older .ptcs formats
        self.maxchanu = int(np.fromfile(f, dtype=np.uint64, count=1)) # maxchanid

        self.nt = int(np.fromfile(f, dtype=np.uint64, count=1)) # nt: number of time points in template

        self.nwavedatabytes, self.wavedata = self.read_wave(f) #TEMPLATE

        self.nwavestdbytes, self.wavestd = self.read_wave(f) #STANDARD DEVIATION
        self.nspikes = int(np.fromfile(f, dtype=np.uint64, count=1)) # nspikes

        self.spikes = np.fromfile(f, dtype=np.uint64, count=self.nspikes)

        # convert from unsigned to signed int for calculating intervals:
        self.spikes = np.asarray(self.spikes, dtype=np.float64)

            
    def read_wave(self, f):
        """Read wavedata/wavestd bytes"""
        # nwavedata/nwavestd bytes, padded:
        nbytes = int(np.fromfile(f, dtype=np.uint64, count=1))
        fp = f.tell()
        count = nbytes // self.nsamplebytes # trunc to ignore any pad bytes
        X = np.fromfile(f, dtype=self.wavedtype, count=count) # wavedata/wavestd (uV)
        if nbytes != 0:
            X.shape = self.nchans, self.nt # reshape
        f.seek(fp + nbytes) # skip any pad bytes
        return nbytes, X

    def rstrip(s, strip):
        """What I think str.rstrip should really do"""
        if s.endswith(strip):
            return s[:-len(strip)] # strip it
        else:
            return s

    def read(self):
        self.nid = self.parse_id()
        with open(self.fname, 'rb') as f:
            self.spikes = np.fromfile(f, dtype=np.int64) # spike timestamps (us)
        self.nspikes = len(self.spikes)

    def read_tsf(self,f):
        pass

class Load_lfp(object):
    """Polytrode clustered spikes file neuron record"""
    
    def __init__(self, fname):

        self.loadLFP(fname)
        self.name = fname

    def loadLFP(self,fname):

        print "Loading LFP"
        self.name=fname
        fin = open(fname, "rb")

        if 'tim' in fname:
            
            self.header = fin.read(16)
            self.iformat = struct.unpack('i',fin.read(4))[0] 
            self.SampleFrequency = struct.unpack('i',fin.read(4))[0] 
            self.n_electrodes = struct.unpack('i',fin.read(4))[0] 
            self.n_vd_samples = struct.unpack('i',fin.read(4))[0] 
            self.vscale_HP = struct.unpack('f',fin.read(4))[0] 
            print "iformat: ", self.iformat
            print self.vscale_HP
            self.vscale_HP = 1.0

            if self.iformat==1002:
                self.Siteloc = np.zeros((2*self.n_electrodes), dtype=np.int16)
                self.Readloc = np.zeros((self.n_electrodes), dtype=np.int32)
                for i in range(self.n_electrodes):
                    self.Siteloc[i*2] = struct.unpack('h', fin.read(2))[0]
                    self.Siteloc[i*2+1] = struct.unpack('h', fin.read(2))[0]
                    self.Readloc[i] = struct.unpack('i', fin.read(4))[0]
            
            self.lfp_sites = self.Siteloc
            self.lfp_sites_positions = self.Readloc

            self.lfp_traces =  np.fromfile(fin, dtype=np.int16, count=self.n_electrodes*self.n_vd_samples)
            self.lfp_traces.shape = self.n_electrodes, self.n_vd_samples

            #Convert data to .lfp format
            self.n_lfp_samples = self.n_vd_samples
            self.n_lfp_records = self.n_vd_samples

            file_name = self.name[:-4]+ '_scaled_notch_multiunit.tsf'

            if False:
                with open(file_name, 'wb') as f:
                    out_SampleFrequency = self.SampleFrequency*2E+1
                    print out_SampleFrequency
                    f.write(self.header)
                    f.write(struct.pack('i', self.iformat))
                    f.write(struct.pack('i', out_SampleFrequency))
                    f.write(struct.pack('i', self.n_electrodes))
                    f.write(struct.pack('i', self.n_vd_samples))
                    f.write(struct.pack('f', self.vscale_HP))

                    for i in range(self.n_electrodes):
                        f.write(struct.pack('h', self.Siteloc[i*2]))
                        f.write(struct.pack('h', self.Siteloc[i*2+1]))
                        f.write(struct.pack('i', i+1))
                        
                    for i in range(self.n_electrodes):
                        data = np.array(self.lfp_traces[i], dtype=np.int16)

                        #Apply butterworth filters - if required
                        fs = self.SampleFrequency
                        lowcut = 50
                        highcut = 150.0
                        data = np.array(butter_bandpass_filter(data, lowcut, highcut, fs, order = 2), dtype=np.int16)
                        
                        #Apply 'notch" butterworth filter
                        #data = np.array(filter.notch(data)[0],dtype=np.int16) # remove 60 Hz mains noise, as for SI calc
                        #data = data*0.1 #Compress amplitudes by factor of 10
                        
                        data.tofile(f)

                    f.write(struct.pack('i', 0)) #Write # of fake spikes
                    f.close()   
                
                print "Exported notch filter, temp compressed lfp: ", file_name
                quit()

        print "Loading LFP"
        self.name=fname
        fin = open(fname, "rb")
        
        if 'nick' in fname:
            #with open(fname,'rb') as fin:
            self.n_electrodes = 10
            self.header = fin.read(24)
            self.surf_filename = fin.read(256)

            self.n_lfp_samples = struct.unpack('i',fin.read(4))[0] 
            self.n_lfp_records = struct.unpack('i',fin.read(4))[0] 
            self.n_vd_samples = self.n_lfp_samples

            self.lfp_traces =  np.fromfile(fin, dtype=np.int16, count=self.n_electrodes*self.n_lfp_samples)
            self.lfp_traces.shape = self.n_electrodes, self.n_lfp_samples                

            #Careful: Nick's LFP may not align with original data; Needs separate loading/consideration
            self.lfp_time_stamps =  np.fromfile(fin, dtype=np.int64, count=self.n_lfp_records)
            self.lfp_time_stamps.shape= self.n_lfp_records/self.n_electrodes, self.n_electrodes
            self.lfp_time_stamps=self.lfp_time_stamps.T/1E+6/60. #Convert from usec to seconds

            self.lfp_sites = np.fromfile(fin, dtype=np.int32, count=10)
            self.lfp_sites_positions = np.fromfile(fin, dtype=np.int32, count=10)

            ##Fix channel mapping for LFP sites
            #print self.lfp_sites
            #self.lfp_sites = [21, 18, 15, 12, 9, 6, 23, 3, 25, 26]
            #channels_2b= Sort1.chanpos[self.lfp_sites].T[1]
            #print channels_2b

            ##Sort LFP Channels by depth for export
            #xy = zip(channels_2b, self.lfp_traces)
            #xy.sort()
            #self.lfp_traces = np.array([x for y, x in xy])
            #channels_2b = [x for x, y in xy]
           
            self.SampleFrequency = 1000 #NEED TO AUTOMATE THIS SOMEHOW FROM NICK'S LFP FILES AND SEV'S DATA

            self.jacked_electrodes=[]

            if False:
                print "Saving temporally compressed lfp"

                #Convert traces to int16
                ecp_total = np.array(self.lfp_traces*10., dtype=np.int16)

                file_name = self.name[:-4]+ '_scaled_notch_multiunit.tsf'
                header = 'Test spike file '
                iformat = 1002

                SampleFrequency = int(1000./self.SampleFrequency)
                vscale_HP = 0.1
                Siteloc = np.zeros((self.n_electrodes,2), dtype=np.int16)

                for i in range(self.n_electrodes):
                    Siteloc[i][0]=0
                    Siteloc[i][1]=i*100

                with open(file_name, 'wb') as f:
                    f.write(header)
                    f.write(struct.pack('i', iformat))
                    if 'nick' in self.name:
                        f.write(struct.pack('i', self.SampleFrequency*100))  #takes ~100ms events to 1ms events
                        #f.write(struct.pack('i', self.SampleFrequency)) 
                        for i in range(self.n_electrodes):
                            Siteloc[i][0]=0
                            Siteloc[i][1]=i*100 #channels_2b[i]
                    else:
                        f.write(struct.pack('i', self.SampleFrequency))
                    f.write(struct.pack('i', self.n_electrodes))
                    f.write(struct.pack('i', self.n_vd_samples))
                    f.write(struct.pack('f', vscale_HP))
                    for i in range(self.n_electrodes):
                        f.write(struct.pack('h', Siteloc[i][0]))
                        f.write(struct.pack('h', Siteloc[i][1]))
                        f.write(struct.pack('i', i+1))
                        
                    for i in range(self.n_electrodes):
                        data = np.array(self.lfp_traces[i]*10., dtype=np.int16)

                        #Apply butterworth filters - if required
                        #fs = self.SampleFrequency
                        #lowcut = 0.1
                        #highcut = 25.0
                        #data = np.array(butter_bandpass_filter(data, lowcut, highcut, fs, order = 2), dtype=np.int16)
                        

                        #Apply 'notch" butterworth filter
                        data = np.array(filter.notch(data)[0],dtype=np.int16) # remove 60 Hz mains noise, as for SI calc
                        #data = data*0.1 #Compress amplitudes by factor of 10
                        
                        data.tofile(f)

                    f.write(struct.pack('i', 0)) #Write # of fake spikes
                    f.close()   
                
                print "Exported notch filter, temp compressed lfp: ", file_name
                quit()
            
        if 'sev' in fname or 'dan' in fname:

            self.header = fin.read(16)
            self.iformat = struct.unpack('i',fin.read(4))[0] 
            self.SampleFrequency = struct.unpack('i',fin.read(4))[0] 
            self.n_electrodes = struct.unpack('i',fin.read(4))[0] 
            self.n_vd_samples = struct.unpack('i',fin.read(4))[0] 
            self.vscale_HP = struct.unpack('f',fin.read(4))[0] 
            print "iformat: ", self.iformat

            if self.iformat==1001:
                self.Siteloc = np.zeros((2*56), dtype=np.int16)
                self.Siteloc = struct.unpack(str(2*56)+'h', fin.read(2*56*2))
            if self.iformat==1002:
                self.Siteloc = np.zeros((2*self.n_electrodes), dtype=np.int16)
                self.Readloc = np.zeros((self.n_electrodes), dtype=np.int32)
                for i in range(self.n_electrodes):
                    self.Siteloc[i*2] = struct.unpack('h', fin.read(2))[0]
                    self.Siteloc[i*2+1] = struct.unpack('h', fin.read(2))[0]
                    self.Readloc[i] = struct.unpack('i', fin.read(4))[0]

            self.ec_traces =  np.fromfile(fin, dtype=np.int16, count=self.n_electrodes*self.n_vd_samples)
            self.ec_traces.shape = self.n_electrodes, self.n_vd_samples

            #Convert data to .lfp format
            self.n_lfp_samples = self.n_vd_samples
            self.n_lfp_records = self.n_vd_samples

            if 'sev' in fname:
                self.lfp_traces = np.flipud(self.ec_traces) #Sev's LFP traces start from the bottom up;

                #MASK OUT PROBLEM CHANNELS
                if 'M72' in fname: #2 electrodes are jacked!
                    self.jacked_electrodes=[26,22]
                    for i in self.jacked_electrodes:
                        self.lfp_traces[i] = self.lfp_traces[i-1] # = np.delete(self.lfp_traces, (i),axis=0)

                    #Load sweeparray 
                    temp_file = '/media/cat/Data1/in_vivo/sev/LGN/M72/1CDS1.SweepTime.mat'
                    mat = scipy.io.loadmat(temp_file)
                    indexes = mat.keys()
                    array = np.array(mat[indexes[1]][0][0][4][0][1]) #Cat: many wrappers on these data structures 
                    self.bright_array= array/float(self.SampleFrequency) #Convert to seconds


                elif 'M73' in fname: #at least 5 electrodes don't look right; also ~#8, but unclear how to remove
                    self.jacked_electrodes = [31,30,29,1,0] #Can remove top and bottom layers
                    for i in self.jacked_electrodes:
                        self.lfp_traces = np.delete(self.lfp_traces, (i),axis=0)
                        self.n_electrodes-=1

                    #Copy channel ; easiest way to remove it's effect on CSD computations
                    self.lfp_traces[7]=self.lfp_traces[6]
                    self.lfp_traces[3]=self.lfp_traces[2]
                    self.jacked_electrodes.extend([7,3])

                    #Load sweeparray 
                    temp_file = '/media/cat/Data1/in_vivo/sev/LGN/M73/1CSD1.SweepTime.mat'
                    mat = scipy.io.loadmat(temp_file)
                    indexes = mat.keys()
                    
                    array = np.array(mat[indexes[1]][0][0][4][0][1]) #Cat: many wrappers on these data structures 
                    self.bright_array= array/float(self.SampleFrequency) #Convert to seconds

                elif 'M74' in fname: 
                    self.jacked_electrodes=[9,5,3]
                    for i in self.jacked_electrodes:
                        self.lfp_traces[i] = self.lfp_traces[i-1] # = np.delete(self.lfp_traces, (i),axis=0)

                    #Load sweeparray 
                    temp_file = '/media/cat/Data1/in_vivo/sev/LGN/M74/CSD1.SweepTime.mat'
                    mat = scipy.io.loadmat(temp_file)
                    indexes = mat.keys()
                    
                    array = np.array(mat[indexes[1]][0][0][4][0][1]) #Cat: many wrappers on these data structures 
                    self.bright_array= array/float(self.SampleFrequency) #Convert to seconds


            if False:
                print "Saving Sergey Rectified (filtered + abs) .tsf files:"

                #Create Recording group
                hdf5file = h5py.File('/media/cat/4TB/in_vivo/sev/M71/CSD3/CSD3_rectified.hdf5','w',libver='latest')

                grp=hdf5file.create_group('HDF5 rectified')
                grp.attrs['recording name']= 'Sev M71 Experiment - Sergey Rectified data'

                ##Load stimulus times - SEV DATA
                file_name = '/media/cat/4TB/in_vivo/sev/M71/CSD3/CSD3.SweepTime.mat'
                mat = scipy.io.loadmat(file_name)
                bright_array = np.array(mat[mat.keys()[1]][0][0][4][0][1])/float(self.SampleFrequency)  
                bright_array = bright_array.T[0] #Select just starting time
                
                header = 'Test spike file '
                iformat = 1002
                vscale_HP = 1.0
                Siteloc = np.zeros((self.n_electrodes,2), dtype=np.int16)
                for i in range(self.n_electrodes):
                    Siteloc[i][0]=0
                    Siteloc[i][1]=i*20

                file_name = self.name[:-4]+ '_0_500Hz.tsf'
                with open(file_name, 'wb') as f:
                    f.write(header)
                    f.write(struct.pack('i', iformat))
                    if 'nick' in self.name:
                        f.write(struct.pack('i', self.SampleFrequency*100))  #20 shrinking factor takes ~100ms events to 5ms events
                    else:
                        f.write(struct.pack('i', self.SampleFrequency))
                    f.write(struct.pack('i', self.n_electrodes))
                    f.write(struct.pack('i', self.n_vd_samples))
                    f.write(struct.pack('f', vscale_HP))
                    for i in range(self.n_electrodes):
                        f.write(struct.pack('h', Siteloc[i][0]))
                        f.write(struct.pack('h', Siteloc[i][1]))
                        f.write(struct.pack('i', i+1))
                    
                    data_array=[]
                    for i in range(self.n_electrodes):
                        fs = self.SampleFrequency
                        lowcut = 0.1
                        highcut = 500.0
                        data = np.abs(butter_bandpass_filter(self.lfp_traces[i], lowcut, highcut, fs, order = 2))
                        np.array(np.abs(data),dtype=np.int16).tofile(f)
                        data_array.append(data)

                    f.write(struct.pack('i', 0)) #Write # of fake spikes
                    f.close()

                #Save low pass rectified trial averaged data:
                stim_array= bright_array
                window = 1 #sec of window to average
                lfp_cumulative = np.zeros((self.n_electrodes,window*self.SampleFrequency), dtype=np.float32)
                for i in range(self.n_electrodes):
                    counter=0
                    for j in stim_array:
                        start=int(j*self.SampleFrequency)
                        end=min(start+window*self.SampleFrequency, self.n_vd_samples) #only search to end of recording
                        lfp_cumulative[i] += data_array[i][start:end]
                lfp_cumulative /= len(stim_array)
                
                grp.create_dataset('0_500Hz',data=np.array(lfp_cumulative,dtype=np.float32))
                grp.attrs['raw_ecp']=  'Trial averaged, rectified ("abs") data, filtered 0-500Hz'

                print "Exported low pass data"


                #**********************************************************
                file_name = self.name[:-4]+ '_500_5000Hz.tsf'
                with open(file_name, 'wb') as f:
                    f.write(header)
                    f.write(struct.pack('i', iformat))
                    if 'nick' in self.name:
                        f.write(struct.pack('i', self.SampleFrequency*100))  #20 shrinking factor takes ~100ms events to 5ms events
                    else:
                        f.write(struct.pack('i', self.SampleFrequency))
                    f.write(struct.pack('i', self.n_electrodes))
                    f.write(struct.pack('i', self.n_vd_samples))
                    f.write(struct.pack('f', vscale_HP))
                    for i in range(self.n_electrodes):
                        f.write(struct.pack('h', Siteloc[i][0]))
                        f.write(struct.pack('h', Siteloc[i][1]))
                        f.write(struct.pack('i', i+1))
                        
                    data_array=[]
                    for i in range(self.n_electrodes):
                        fs = self.SampleFrequency 
                        lowcut = 500.0
                        highcut = 5000.0
                        data = np.abs(butter_bandpass_filter(self.lfp_traces[i], lowcut, highcut, fs, order = 2))
                        np.array(np.abs(data),dtype=np.int16).tofile(f)
                        data_array.append(data)

                    f.write(struct.pack('i', 0)) #Write # of fake spikes
                    f.close()   

                #Save low pass rectified trial averaged data:
                stim_array= bright_array
                window = 1 #sec of window to average
                lfp_cumulative = np.zeros((self.n_electrodes,window*self.SampleFrequency), dtype=np.float32)
                for i in range(self.n_electrodes):
                    counter=0
                    for j in stim_array:
                        start=int(j*self.SampleFrequency)
                        end=min(start+window*self.SampleFrequency, self.n_vd_samples) #only search to end of recording
                        lfp_cumulative[i] += data_array[i][start:end]
                lfp_cumulative /= len(stim_array)
                
                grp.create_dataset('500-5000Hz',data=np.array(lfp_cumulative,dtype=np.float32))
                grp.attrs['raw_ecp']=  'Trial averaged, rectified ("abs") data, filtered 500_5000Hz'

                print "Exported high pass data"
                

                file_name = self.name[:-4]+ '_300_500Hz.tsf'
                with open(file_name, 'wb') as f:
                    f.write(header)
                    f.write(struct.pack('i', iformat))
                    if 'nick' in self.name:
                        f.write(struct.pack('i', self.SampleFrequency*100))  #20 shrinking factor takes ~100ms events to 5ms events
                    else:
                        f.write(struct.pack('i', self.SampleFrequency))
                    f.write(struct.pack('i', self.n_electrodes))
                    f.write(struct.pack('i', self.n_vd_samples))
                    f.write(struct.pack('f', vscale_HP))
                    for i in range(self.n_electrodes):
                        f.write(struct.pack('h', Siteloc[i][0]))
                        f.write(struct.pack('h', Siteloc[i][1]))
                        f.write(struct.pack('i', i+1))
                        
                    data_array=[]
                    for i in range(self.n_electrodes):
                        fs = self.SampleFrequency 
                        lowcut = 300.0
                        highcut = 500.0
                        data = np.abs(butter_bandpass_filter(self.lfp_traces[i], lowcut, highcut, fs, order = 2))
                        np.array(data,dtype=np.int16).tofile(f)
                        data_array.append(data)

                    f.write(struct.pack('i', 0)) #Write # of fake spikes
                    f.close()                   
                
                #Save low pass rectified trial averaged data:
                stim_array= bright_array
                window = 1 #sec of window to average
                lfp_cumulative = np.zeros((self.n_electrodes,window*self.SampleFrequency), dtype=np.float32)
                for i in range(self.n_electrodes):
                    counter=0
                    for j in stim_array:
                        start=int(j*self.SampleFrequency)
                        end=min(start+window*self.SampleFrequency, self.n_vd_samples) #only search to end of recording
                        lfp_cumulative[i] += data_array[i][start:end]
                lfp_cumulative /= len(stim_array)
                
                grp.create_dataset('300_500Hz',data=np.array(lfp_cumulative,dtype=np.float32))
                grp.attrs['raw_ecp']=  'Trial averaged, rectified ("abs") data, filtered 300-500Hz'
                
                print "Exported band pass data"
                
                #*************************************************************
                file_name = self.name[:-4]+ '.tsf'
                with open(file_name, 'wb') as f:
                    f.write(header)
                    f.write(struct.pack('i', iformat))
                    if 'nick' in self.name:
                        f.write(struct.pack('i', self.SampleFrequency*100))  #20 shrinking factor takes ~100ms events to 5ms events
                    else:
                        f.write(struct.pack('i', self.SampleFrequency))
                    f.write(struct.pack('i', self.n_electrodes))
                    f.write(struct.pack('i', self.n_vd_samples))
                    f.write(struct.pack('f', vscale_HP))
                    for i in range(self.n_electrodes):
                        f.write(struct.pack('h', Siteloc[i][0]))
                        f.write(struct.pack('h', Siteloc[i][1]))
                        f.write(struct.pack('i', i+1))
                        
                    data_array=[]
                    for i in range(self.n_electrodes):
                        data = np.abs(self.lfp_traces[i])
                        np.array(data,dtype=np.int16).tofile(f)
                        data_array.append(data)

                    f.write(struct.pack('i', 0)) #Write # of fake spikes
                    f.close()  

                #Save low pass rectified trial averaged data:
                stim_array= bright_array
                window = 1 #sec of window to average
                lfp_cumulative = np.zeros((self.n_electrodes,window*self.SampleFrequency), dtype=np.float32)
                for i in range(self.n_electrodes):
                    counter=0
                    for j in stim_array:
                        start=int(j*self.SampleFrequency)
                        end=min(start+window*self.SampleFrequency, self.n_vd_samples) #only search to end of recording
                        lfp_cumulative[i] += data_array[i][start:end]
                lfp_cumulative /= len(stim_array)
                
                grp.create_dataset('Unfiltered',data=np.array(lfp_cumulative,dtype=np.float32))
                grp.attrs['raw_ecp']=  'Trial averaged, rectified ("abs") data, unfiltered'
                
                print "Exported unfiltered data"

                hdf5file.close()
                
                quit()



            else:
                self.lfp_traces =  self.ec_traces
                self.jacked_electrodes=[]

            if 'dan' in fname:
                file_name = '/home/cat/neuron/in_vivo/dan/M149873/2015-02-06_13-02-32_flash_CSD/data/stim_times.txt'
                self.bright_array= np.loadtxt(file_name,dtype=np.float32) #Convert to seconds

            #For other than Nick's data, the LFP time stamps for each channel are the same; i.e. need only one array 
            self.lfp_time_stamps = np.arange(0,self.n_vd_samples,1./60., dtype=np.float32)   #convert directly to seconds

            self.lfp_sites = self.Siteloc 
            if 'truth' in fname:
                self.lfp_sites_positions = self.Siteloc  #This is just so I could load an old simulation file
            else:
                self.lfp_sites_positions = self.Siteloc 

        fin.close()

class Loadcsv(object):

    def __init__(self, fname, tsf, work_dir, csv_version):

        self.fname = fname
        self.loadSpikes(fname, tsf, work_dir, csv_version)


    def loadSpikes(self, fname, tsf, work_dir, csv_version):

        #Dan's data and spikesortingtest.com data structure
        #NB: Ensure that 3 decimal places spiketimes (i.e. ms precision) is sufficient
        if csv_version==2: 

            f = open(fname, "r")
            data_file = np.genfromtxt(f, delimiter=',', dtype=np.float32) #, usecols=(0,))
            f.close()

            #Dan's sorted .csv file switches the spiketimes and unit id columns 
            if ('dan' in self.fname) or ('nickst' in self.fname):
                spiketimes=data_file[:,1] 
                spiketimes= np.array(spiketimes*tsf.SampleFrequency, dtype=np.float32) #Convert spiketimes to timesteps
            
                spike_id=np.array(data_file[:,0],dtype=np.int32)

                print spiketimes
                print spike_id

            elif ('pierre' in self.fname):
                f = open(fname, "r")
                data_file = np.genfromtxt(f, delimiter=' ', dtype=np.float32) #, usecols=(0,))
                f.close()

                spiketimes=data_file[:,0] 
                spiketimes= np.array(spiketimes*tsf.SampleFrequency*1E-3, dtype=np.float32) #Convert spiketimes to timesteps

                spike_id=np.array(data_file[:,1],dtype=np.int32)
                
                print spiketimes
                print spike_id
                #quit()

            elif ('martin' in self.fname):
                f = open(fname, "r")
                data_file = np.genfromtxt(f, delimiter=',', dtype=np.float32) #, usecols=(0,))
                f.close()

                spiketimes=data_file[:,0] 
                spiketimes= np.array(spiketimes*tsf.SampleFrequency, dtype=np.float32) #Convert spiketimes to timesteps

                spike_id=np.array(data_file[:,1],dtype=np.int32)
                
                print spiketimes
                print spike_id
                #quit()

            elif ('peter' in self.fname):
                f = open(fname, "r")
                data_file = np.genfromtxt(f, delimiter=',', dtype=np.float32) #, usecols=(0,))
                f.close()

                spiketimes=data_file[:,0] 
                spiketimes= np.array(spiketimes*tsf.SampleFrequency, dtype=np.float32) #Convert spiketimes to timesteps

                spike_id=np.array(data_file[:,1],dtype=np.int32)
                
                print spiketimes
                print spike_id


            #Load max-chan data if available
            if len(data_file[0])>2:
                self.maxchan= data_file[:,2]
            else:
                self.maxchan= np.zeros(len(data_file),dtype=np.float32)

            #Map Klustakwik discountinuos unit ids onto 0 based incremental ids
            unique_ids = np.unique(spike_id)

            #print "Parsing Dan .csv file -> assigning to units"
            self.units=[]
            for i in range(len(unique_ids)):
                self.units.append([])

            for i in range(len(spiketimes)):
                self.units[np.where(unique_ids==spike_id[i])[0][0]].append(spiketimes[i])

            self.n_units=len(self.units)
            #print "No. of units in .csv: ", self.n_units

            if (os.path.exists(fname[0:-4]+'_ptps.csv')==False):
                #Compute PTP values from original .tsf file; needed for .csv sorted files
                self.ptp=np.zeros((self.n_units), dtype=np.float32)
                self.size=np.zeros((self.n_units), dtype=np.float32)
                for i in range(self.n_units):
                    self.size[i]=len(self.units[i])

                for i in range(self.n_units):
                    tsf.compute_ptp(self.units[i], tsf.n_electrodes, tsf.ec_traces, tsf.SampleFrequency, tsf.vscale_HP)
                    self.ptp[i]=tsf.ptp
                    self.maxchan[i]=tsf.maxchan
                    print "Unit: ", i , " ptp: ", self.ptp[i], " size: ", self.size[i], " max ch: ", tsf.maxchan
            else:
                
                self.ptp = np.loadtxt(fname[:-4]+'_ptps.csv', dtype=np.float32, delimiter=",")
                self.maxchan = np.loadtxt(fname[:-4]+'_maxch.csv', dtype=np.int32, delimiter=",")
                self.size = np.loadtxt(fname[:-4]+'_size.csv', dtype=np.int32, delimiter=",")       


            ##Order units by PTP! Important for later steps
            ptps = self.ptp
            ptps_indexes = np.argsort(ptps)[::-1] #Re Order it from highest to lowest
            
            copy_units = np.array(self.units).copy()
            copy_ptp=np.array(self.ptp).copy()
            copy_maxchan=np.array(self.maxchan).copy()
            copy_size=np.array(self.size).copy()
           
            #Re arrange all sorted data by PTP order
            self.units=[]
            self.ptp=[]
            self.maxchan=[]
            self.size=[]
            for i in range(self.n_units):
                self.units.append(copy_units[ptps_indexes[i]])
                self.ptp.append(copy_ptp[ptps_indexes[i]])
                self.maxchan.append(copy_maxchan[ptps_indexes[i]])
                self.size.append(copy_size[ptps_indexes[i]])

            #Save ptp ordered support files - if not already there; 
            #TODO: include all of this as part of above conditional
            #PROBLEM IS THAT self.units is being loaded from original file everytime and the order must be recomupted otherwise
            #NEED to re-save the original .csv file - which is not a good idea
            if (os.path.exists(fname[0:-4]+'_ptps.csv')==False):
                np.savetxt(fname[:-4]+'_ptps.csv', self.ptp, delimiter=",")
                np.savetxt(fname[:-4]+'_maxch.csv', self.maxchan, fmt='%i', delimiter=",")
                np.savetxt(fname[:-4]+'_size.csv', self.size, delimiter=",")
                
                
            self.n_sorted_spikes = [None]*self.n_units
            for k in range(self.n_units):
                self.n_sorted_spikes[k] = len(self.units[k])


        elif csv_version==1:
            f = open(fname, "r")
            spiketimes = np.genfromtxt(f, delimiter=',', dtype=float32, usecols=(0,))
            f.close()
            f = open(fname, "r")
            spikeids =  np.genfromtxt(f, delimiter=',', dtype=int32, usecols=(1,))

            self.n_sorted_spikes = len(spiketimes)
            self.n_units = max(spikeids)
            self.units=[[] for x in range(self.n_units)]

            for k in range(len(spiketimes)):
                self.units[spikeids[k]-1].extend([spiketimes[k]])

            self.n_sorted_spikes = [None]*self.n_units
            for k in range(self.n_units):
                self.n_sorted_spikes[k] = len(self.units[k])

            #James spiketimes need to be scaled as below
            #Scale spiketimes; DO THIS IN NUMPY?! couldn't get it to work earlier.
            for k in range(self.n_units):
                self.units[k]=[i * tsf.SampleFrequency for i in self.units[k]] 

        elif csv_version==3:
            f = open(fname, "r")
            spiketimes = np.genfromtxt(f, delimiter=',', dtype=float32, usecols=(0,))
            f.close()
            f = open(fname, "r")
            spikeids =  np.genfromtxt(f, delimiter=',', dtype=int32, usecols=(1,))

            self.n_sorted_spikes = len(spiketimes)
            self.n_units = max(spikeids)
            self.units=[[] for x in range(self.n_units)]

            for k in range(len(spiketimes)):
                self.units[spikeids[k]-1].extend([spiketimes[k]])

            self.n_sorted_spikes = [None]*self.n_units
            for k in range(self.n_units):
                self.n_sorted_spikes[k] = len(self.units[k])

            #James spiketimes need to be scaled as below
            #Scale spiketimes; DO THIS IN NUMPY?! couldn't get it to work earlier.
            for k in range(self.n_units):
                self.units[k]=[i * tsf.SampleFrequency*1E-9 for i in self.units[k]] 

        elif csv_version==4:

            f = open(fname, "r")
            data_file = np.genfromtxt(f, delimiter=',', dtype=int32) #, usecols=(0,))
            f.close()

            spiketimes=data_file[:,0]
            spike_id1=data_file[:,1]
            spike_id2=data_file[:,2]
            self.units=[]

            for i in range(len(np.unique(spike_id1))-1):  #Remove CK's '-1' index meaning no match
                self.units.append([])

            print "Parsing CK .csv file -> assigning to units"
            for i in range(len(spiketimes)):
                if spike_id1[i]!= -1: 
                    self.units[spike_id1[i]-1].append(spiketimes[i])
                if spike_id2[i]!= -1: 
                    self.units[spike_id2[i]-1].append(spiketimes[i])

            self.n_sorted_spikes = len(spiketimes)
            self.n_units = len(np.unique(spike_id1))-1

            self.n_sorted_spikes = [None]*self.n_units
            for k in range(self.n_units):
                self.n_sorted_spikes[k] = len(self.units[k])

            #James spiketimes need to be scaled as below
            #Scale spiketimes; DO THIS IN NUMPY?! couldn't get it to work earlier.
            #for k in range(self.n_units):
            #    self.units[k]=[i * tsf.SampleFrequency*1E-9 for i in self.units[k]] 

        elif csv_version==5:

            f = open(fname, "r")
            data_file = np.genfromtxt(f, delimiter=',', dtype=int32) #, usecols=(0,))
            f.close()

            spiketimes=data_file[:,0]
            spike_id1=data_file[:,2]
            spike_id2=data_file[:,3]
            max_chan1=data_file[:,4]
            max_chan2=data_file[:,5]

            #print max_chan1
            #quit()
            self.units=[]
            for i in range(len(np.unique(spike_id1))-1):  #Remove CK's '-1' index meaning no match
                self.units.append([])

            print "Parsing CK .csv file -> assigning to units"
            self.maxchan= np.zeros(len(self.units), dtype=np.int32)

            for i in range(len(spiketimes)):
                if spike_id1[i]!= -1:
                    self.units[spike_id1[i]-1].append(spiketimes[i])
                    self.maxchan[spike_id1[i]-1]=max_chan1[i]
                if spike_id2[i]!= -1:
                    self.units[spike_id2[i]-1].append(spiketimes[i])

            self.n_sorted_spikes = len(spiketimes)
            self.n_units = len(np.unique(spike_id1))-1

            self.n_sorted_spikes = [None]*self.n_units
            for k in range(self.n_units):
                self.n_sorted_spikes[k] = len(self.units[k])

            #James spiketimes need to be scaled as below
            #Scale spiketimes; DO THIS IN NUMPY?! couldn't get it to work earlier.
            #for k in range(self.n_units):
            #    self.units[k]=[i * tsf.SampleFrequency*1E-9 for i in self.units[k]]


        elif csv_version==6:  #CK's 4 unit ID's file;

            f = open(fname, "r")
            data_file = np.genfromtxt(f, delimiter=',', dtype=int32) #, usecols=(0,))
            f.close()

            spiketimes=data_file[:,1] #Use crosstime values, not initial timestamps; might be more accurate.
            spike_id1=data_file[:,2]
            spike_id2=data_file[:,3]
            spike_id3=data_file[:,4]
            spike_id4=data_file[:,5]
            
            max_chan1=data_file[:,6]
            max_chan2=data_file[:,7]
            max_chan3=data_file[:,8]
            max_chan4=data_file[:,9]

            #print max_chan1
            #quit()
            self.units=[]
            for i in range(len(np.unique(spike_id1))-1):  #Remove 1 from total # units for CK's '-1' (i.e. no match index)
                self.units.append([])

            print "Parsing CK .csv file -> assigning to units"
            self.maxchan= np.zeros(len(self.units), dtype=np.int32)

            for i in range(len(spiketimes)):
                if spike_id1[i]!= -1:
                    self.units[spike_id1[i]-1].append(spiketimes[i])
                    self.maxchan[spike_id1[i]-1]=max_chan1[i]
                if spike_id2[i]!= -1:
                    self.units[spike_id2[i]-1].append(spiketimes[i])
                if spike_id3[i]!= -1:
                    self.units[spike_id3[i]-1].append(spiketimes[i])
                if spike_id4[i]!= -1:
                    self.units[spike_id4[i]-1].append(spiketimes[i])

            self.n_sorted_spikes = len(spiketimes)
            self.n_units = len(np.unique(spike_id1))-1

            self.n_sorted_spikes = [None]*self.n_units
            for k in range(self.n_units):
                self.n_sorted_spikes[k] = len(self.units[k])

        elif csv_version==7:  #CK's 4 unit ID's file;

            f = open(fname, "r")
            data_file = np.genfromtxt(f, delimiter=',', dtype=int32) #, usecols=(0,))
            f.close()

            spiketimes=data_file[:,1] #Use crosstime values, not initial timestamps; might be more accurate.
            spike_id1=data_file[:,2]
            spike_id2=data_file[:,3]
            spike_id3=data_file[:,4]
            spike_id4=data_file[:,5]
            spike_id5=data_file[:,6]
            
            max_chan1=data_file[:,7]
            max_chan2=data_file[:,8]
            max_chan3=data_file[:,9]
            max_chan4=data_file[:,10]
            max_chan5=data_file[:,11]

            self.units=[]
            for i in range(len(np.unique(spike_id1))-1):  #Remove 1 from total # units for CK's '-1' (i.e. no match index)
                self.units.append([])

            print "Parsing CK .csv file -> assigning to units"
            self.maxchan= np.zeros(len(self.units), dtype=np.int32)

            for i in range(len(spiketimes)):
                if spike_id1[i]!= -1:
                    self.units[spike_id1[i]-1].append(spiketimes[i])
                    self.maxchan[spike_id1[i]-1]=max_chan1[i]
                if spike_id2[i]!= -1:
                    self.units[spike_id2[i]-1].append(spiketimes[i])
                if spike_id3[i]!= -1:
                    self.units[spike_id3[i]-1].append(spiketimes[i])
                if spike_id4[i]!= -1:
                    self.units[spike_id4[i]-1].append(spiketimes[i])
                if spike_id5[i]!= -1:
                    self.units[spike_id4[i]-1].append(spiketimes[i])

            self.n_sorted_spikes = len(spiketimes)
            self.n_units = len(np.unique(spike_id1))-1

            self.n_sorted_spikes = [None]*self.n_units
            for k in range(self.n_units):
                self.n_sorted_spikes[k] = len(self.units[k])


        #********************COMPUTE UNIT PTP and Maxchan from original .tsf file**********************
        if (os.path.exists(fname[0:-4]+'_ptps.csv')==False):
            #Compute PTP values from original .tsf file; needed for .csv sorted files
            self.ptp=np.zeros((self.n_units), dtype=np.float32)
            self.size=np.zeros((self.n_units), dtype=np.float32)
            for i in range(self.n_units):
                self.size[i]=len(self.units[i])

            for i in range(self.n_units):
                #print "Computing ptp for unit: ", i
                tsf.compute_ptp(self.units[i], tsf.n_electrodes, tsf.ec_traces, tsf.SampleFrequency, tsf.vscale_HP)
                self.ptp[i]=tsf.ptp
                self.maxchan[i]=tsf.maxchan
                print "Unit: ", i , " ptp: ", self.ptp[i], " size: ", self.size[i], " max ch: ", tsf.maxchan

            np.savetxt(fname[:-4]+'_ptps.csv', self.ptp, delimiter=",")
            np.savetxt(fname[:-4]+'_maxch.csv', self.maxchan, fmt='%i', delimiter=",")
            np.savetxt(fname[:-4]+'_size.csv', self.size, delimiter=",")

        else:
            self.ptp = np.loadtxt(fname[:-4]+'_ptps.csv', dtype=np.float32, delimiter=",")
            self.maxchan = np.loadtxt(fname[:-4]+'_maxch.csv', dtype=np.int32, delimiter=",")
            self.size = np.loadtxt(fname[:-4]+'_size.csv', dtype=np.int32, delimiter=",")            
            
            #print self.ptp
            #print self.maxchan
            #print self.size




def Plot_LFP(lfp): #,Sort1):

    ax = plt.subplot(1,1,1) #Can't plot stimulus information;

    title_string = " ".join(str(i) for i in lfp.jacked_electrodes)
    plt.title('LFP (bad electrodes: '+ title_string + ')')

    #PLOT SWEEP TIME from .mat DATA
    if 'sev' in lfp.name:
        for i in range(len(lfp.bright_array)):
            p = ax.axvspan(lfp.bright_array[i,0], lfp.bright_array[i,1], facecolor='black', alpha=0.25)

    if 'dan' in lfp.name:
        for i in range(len(lfp.bright_array)/2):
            p = ax.axvspan(lfp.bright_array[i*2,0], lfp.bright_array[i*2,1], facecolor='black', alpha=0.25)
            p = ax.axvspan(lfp.bright_array[i*2+1,0], lfp.bright_array[i*2+1,1], facecolor='red', alpha=0.25)


    #if 'nick' in lfp.name:
        #for i in range(Sort1.n_units):
            #print "printing unit: ", i
            #print "# spikes: ", len(Sort1.units[i])
            #x = np.array(Sort1.units[i])/1000.
            ##print x
            #ax.vlines(x, 100, -1000, colors='red', linewidth=1)

    showing_start = 0
    showing_end = 10
    showing_length = showing_end-showing_start

    #showing_length = min(showing_length,len(lfp.lfp_traces[0])/lfp.SampleFrequency)  #Seconds of display

    if 'nick' in lfp.name:  showing_length = len(lfp.lfp_traces[0])/lfp.SampleFrequency

    print "Total: ", len(lfp.lfp_traces[0])/lfp.SampleFrequency/60., " minutes (Showing from: ", showing_start, 
    " to ", showing_end, " seconds)."

    x = np.arange(0,len(lfp.lfp_traces[0]),1.)/lfp.SampleFrequency

    if True:
        #Plot extracellular traces - LFPs
        for i in range(lfp.n_electrodes):
            plt.plot(x[showing_start*lfp.SampleFrequency:showing_end*lfp.SampleFrequency],
            0.0025*lfp.lfp_traces[i,showing_start*lfp.SampleFrequency:showing_end*lfp.SampleFrequency]-i, 
            'r-', color='black',linewidth=1)
     
        ax.set_ylim(top=2, bottom = -lfp.n_electrodes-2)
        ax.set_xlim(left=showing_start, right = showing_end)

        ax.set_xlabel('Seconds', fontsize=17)
        ax.set_ylabel('', fontsize=17)
        ax.set_yticks(np.arange(0,-lfp.n_electrodes, -1.0))

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        plt.show()

    #Save depth probe data
    if False:
        #Convert traces to int16
        ecp_total = np.array(lfp.lfp_traces, dtype=np.int16)

        file_name = lfp.name+ '_scaled.tsf'
        header = 'Test spike file '
        iformat = 1002

        SampleFrequency = int(1000./lfp.SampleFrequency)
        vscale_HP = 1.0
        Siteloc = np.zeros((lfp.n_electrodes,2), dtype=np.int16)

        for i in range(lfp.n_electrodes):
            Siteloc[i][0]=0
            Siteloc[i][1]=i*20

        with open(file_name, 'wb') as f:
            f.write(header)
            f.write(struct.pack('i', iformat))
            if 'nick' in lfp.name:
                f.write(struct.pack('i', lfp.SampleFrequency*20))
            else:
                f.write(struct.pack('i', lfp.SampleFrequency))
            f.write(struct.pack('i', lfp.n_electrodes))
            f.write(struct.pack('i', lfp.n_vd_samples))
            f.write(struct.pack('f', vscale_HP))
            for i in range(lfp.n_electrodes):
                f.write(struct.pack('h', Siteloc[i][0]))
                f.write(struct.pack('h', Siteloc[i][1]))
                f.write(struct.pack('i', i))
            for i in range(lfp.n_electrodes):
                ecp_total[i].tofile(f)
            f.write(struct.pack('i', 0)) #Write # of fake spikes
            f.close()    


def Plot_CSD(lfp):

    if 'sev' in lfp.name:
        ax = plt.subplot(1,2,1)
    else:
        ax = plt.subplot(1,1,1) #Can't plot stimulus information;

    title_string = " ".join(str(i) for i in lfp.jacked_electrodes)
    plt.title('LFP + CSD (bad electrodes: '+ title_string + ')')

    #PLOT SWEEP TIME from .mat DATA
    if 'sev' in lfp.name:
        for i in range(len(lfp.bright_array)):
            p = ax.axvspan(lfp.bright_array[i,0], lfp.bright_array[i,1], facecolor='black', alpha=0.25)

    showing_start = 0
    showing_end = 10
    showing_length = showing_end-showing_start

    if 'nick' in lfp.name:  showing_length = len(lfp.lfp_traces[0])/lfp.SampleFrequency

    print "Total: ", len(lfp.lfp_traces[0])/lfp.SampleFrequency/60., " minutes (Showing from: ", showing_start, 
    " to ", showing_end, " seconds)."

    x = np.arange(0,len(lfp.lfp_traces[0]),1.)/lfp.SampleFrequency
    #Plot extracellular traces - LFPs
    if 'sev' in lfp.name: 
        for i in range(lfp.n_electrodes):
            ax.plot(x[showing_start*lfp.SampleFrequency:showing_end*lfp.SampleFrequency],
            0.0025*lfp.lfp_traces[i,showing_start*lfp.SampleFrequency:showing_end*lfp.SampleFrequency]-i, 
            'r-', color='black',linewidth=1)

    lfp_array = lfp.lfp_traces #[:,0:showing_length*lfp.SampleFrequency]

    ##FILTERING OPTION
    #fs = lfp.SampleFrequency 
    #lowcut = 0.1
    #highcut = 100
    #lfp_array = butter_bandpass_filter(lfp_array, lowcut, highcut, fs, order = 2)

    csd = np.diff(lfp_array, n=2, axis=0)

    #Cumulative CSD and LFP: Sum chunks triggered off stimulus start;
    window = 0.5 #Window to be considered/summed

    #CUMULATIVE LFP
    print "Computing cumulative LFP"
    lfp_cumulative = np.zeros((lfp.n_electrodes,window*lfp.SampleFrequency), dtype=np.float32)
    if 'sev' in lfp.name:
        for i in range(lfp.n_electrodes-2):
            for j in range(len(lfp.bright_array)):
                start=int(lfp.bright_array[j,0]*lfp.SampleFrequency)
                end=start+window*lfp.SampleFrequency
                lfp_cumulative[i] += lfp_array[i][start:end]
        lfp_cumulative /= len(lfp.bright_array)

    if 'dan' in lfp.name:
        for i in range(lfp.n_electrodes-2):
            for j in range(len(lfp.bright_array)/2):
                start=int(lfp.bright_array[j*2,0]*lfp.SampleFrequency)
                end=start+window*lfp.SampleFrequency
                lfp_cumulative[i] += lfp_array[i][start:end]
        lfp_cumulative /= len(lfp.bright_array)        

    if False:
        scipy.io.savemat(lfp.name+'.mat', mdict={'arr': lfp_cumulative})
        quit()

    #CUMULATIVE CSD
    print "Computing cumulative CSD"
    csd_cumulative = np.zeros((lfp.n_electrodes-2,window*lfp.SampleFrequency), dtype=np.float32)
    if 'sev' in lfp.name:
        for i in range(lfp.n_electrodes-2):
            for j in range(len(lfp.bright_array)):
                start=int(lfp.bright_array[j,0]*lfp.SampleFrequency)
                end=start+window*lfp.SampleFrequency
                csd_cumulative[i] += csd[i][start:end]
        csd_cumulative /= len(lfp.bright_array)


    if 'dan' in lfp.name:
        for i in range(lfp.n_electrodes-2):
            for j in range(len(lfp.bright_array)/2):
                start=int(lfp.bright_array[j*2,0]*lfp.SampleFrequency)
                end=start+window*lfp.SampleFrequency
                csd_cumulative[i] += csd[i][start:end]
        csd_cumulative /= len(lfp.bright_array)


    ax.set_ylim(top=2, bottom = -lfp.n_electrodes-2)
    ax.set_xlim(left=0, right = showing_length)

    ax.set_xlabel('Seconds', fontsize=17)
    ax.set_ylabel('', fontsize=17)
    ax.set_yticks(np.arange(0,-lfp.n_electrodes, -1.0))

    csd = np.flipud(csd) #The "image" below is upside down, needs to be inverted;
    csd_cumulative = np.flipud(csd_cumulative)

    if 'sev' in lfp.name:
        print "Plotting full-csd"
        im = ax.imshow(csd, cmap=plt.get_cmap('jet'), interpolation='sinc',
                vmin=-250, vmax=250,extent=[0,showing_length,-1,-lfp.n_electrodes+1],aspect='auto',alpha=0.5)

    #fig.colorbar(im)

    if 'dan' in lfp.name:
        print "Plotting sum-csd"
        im = ax.imshow(csd_cumulative, cmap=plt.get_cmap('jet'), interpolation='sinc',
                       vmin=-250, vmax=250,extent=[0,window,-1,-lfp.n_electrodes+1],aspect='auto',alpha=0.5)

        #Plot extracellular traces - LFPs
        for i in range(lfp.n_electrodes):
            ax.plot(x[0:window*lfp.SampleFrequency],0.002*lfp_cumulative[i,0:window*lfp.SampleFrequency]-i, 
            'r-', color='black',linewidth=.25)
        ax.set_ylim(top=2, bottom = -lfp.n_electrodes-2)
        ax.set_yticks(np.arange(0,-lfp.n_electrodes, -1.0))
        ax.set_xlim(left=0, right = window)

    if 'sev' in lfp.name:
        print "Plotting cumulative-csd"
        ax2 = plt.subplot(1, 2, 2)

        plt.title('Average LFP/CSD in 500ms post stimulus')

        im = ax2.imshow(csd_cumulative, cmap=plt.get_cmap('jet'), interpolation='sinc',
                       vmin=-250, vmax=250,extent=[0,window,-1,-lfp.n_electrodes+1],aspect='auto',alpha=0.5)

        #Plot extracellular traces - LFPs
        for i in range(lfp.n_electrodes):
            plt.plot(x[0:window*lfp.SampleFrequency],0.002*lfp_cumulative[i,0:window*lfp.SampleFrequency]-i, 
            'r-', color='black',linewidth=.25)

        ax2.set_ylim(top=2, bottom = -lfp.n_electrodes-2)
        ax2.set_yticks(np.arange(0,-lfp.n_electrodes, -1.0))
        ax2.set_xlim(left=0, right = window)

    #mng = plt.get_current_fig_manager()
    #mng.frame.Maximize(True)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    plt.show()


    #SLIDER - NOT REQUIRED
    #axcolor = 'lightgoldenrodyellow'
    #axfreq = plt.axes([0.5, 0.02, 0.3, 0.03], axisbg=axcolor)
    ##axamp  = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

    #sfreq = Slider(axfreq, 'Time Slider', 0., int(x[-1]), valinit=0)
    ##samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)

    #axslider = plt.axes([0.1, 0.02, 0.3, 0.03], axisbg=axcolor)
    #offset = Slider(axslider, 'Window Width', 0., 10, valinit=0)

    #def update(val):
        ##amp = samp.val
        #freq = sfreq.val
        
        ##l.set_ydata(amp*np.sin(2*np.pi*freq*t))
        #fig.canvas.draw_idle()
        ##for i in range(len(labels)):
        #ax.set_xlim(sfreq.val+0, sfreq.val+offset.val)  
            ##plt.draw()
        ##plt.xlabel('Time (ms, dt: '+str(sample_rate)+' ms)')        
    #sfreq.on_changed(update)


def psd(self, t0=None, t1=None, f0=0.2, f1=110, p0=None, p1=None, chanis=-1,
            width=None, tres=None, xscale='log', figsize=(5, 5)):
    """Plot power spectral density from t0 to t1 in sec, from f0 to f1 in Hz, and clip
    power values from p0 to p1 in dB, based on channel index chani of LFP data. chanis=0
    uses most superficial channel, chanis=-1 uses deepest channel. If len(chanis) > 1,
    take mean of specified chanis. width and tres are in sec."""
    uns = get_ipython().user_ns
    self.get_data()
    ts = self.get_tssec() # full set of timestamps, in sec
    if t0 == None:
        t0, t1 = ts[0], ts[-1] # full duration
    if t1 == None:
        t1 = t0 + 10 # 10 sec window
    if width == None:
        width = uns['LFPWIDTH'] # sec
    if tres == None:
        tres = uns['LFPTRES'] # sec
    assert tres <= width
    NFFT = intround(width * self.sampfreq)
    noverlap = intround(NFFT - tres * self.sampfreq)
    t0i, t1i = ts.searchsorted((t0, t1))
    #ts = ts[t0i:t1i] # constrained set of timestamps, in sec
    data = self.data[:, t0i:t1i] # slice data
    f = pl.figure(figsize=figsize)
    a = f.add_subplot(111)
    if iterable(chanis):
        data = data[chanis].mean(axis=0) # take mean of data on chanis
    else:
        data = data[chanis] # get single row of data at chanis
    #data = filter.notch(data)[0] # remove 60 Hz mains noise
    # convert data from uV to mV. I think P is in mV^2?:
    P, freqs = mpl.mlab.psd(data/1e3, NFFT=NFFT, Fs=self.sampfreq, noverlap=noverlap)
    # keep only freqs between f0 and f1:
    if f0 == None:
        f0 = freqs[0]
    if f1 == None:
        f1 = freqs[-1]
    lo, hi = freqs.searchsorted([f0, f1])
    P, freqs = P[lo:hi], freqs[lo:hi]
    # check for and replace zero power values (ostensibly due to gaps in recording)
    # before attempting to convert to dB:
    zis = np.where(P == 0.0) # row and column indices where P has zero power
    if len(zis[0]) > 0: # at least one hit
        P[zis] = np.finfo(np.float64).max # temporarily replace zeros with max float
        minnzval = P.min() # get minimum nonzero value
        P[zis] = minnzval # replace with min nonzero values
    P = 10. * np.log10(P) # convert power to dB wrt 1 mV^2?
    # for better visualization, clip power values to within (p0, p1) dB
    if p0 != None:
        P[P < p0] = p0
    if p1 != None:
        P[P > p1] = p1
    #self.P = P
    a.plot(freqs, P, 'k-')
    # add SI frequency band limits:
    LFPSILOWBAND, LFPSIHIGHBAND = uns['LFPSILOWBAND'], uns['LFPSIHIGHBAND']
    a.axvline(x=LFPSILOWBAND[0], c='r', ls='--')
    a.axvline(x=LFPSILOWBAND[1], c='r', ls='--')
    a.axvline(x=LFPSIHIGHBAND[0], c='b', ls='--')
    a.axvline(x=LFPSIHIGHBAND[1], c='b', ls='--')
    a.axis('tight')
    a.set_xscale(xscale)
    a.set_xlabel("frequency (Hz)")
    a.set_ylabel("power (dB)")
    #titlestr = lastcmd()
    #gcfm().window.setWindowTitle(titlestr)
    a.set_title(titlestr)
    a.text(0.998, 0.99, '%s' % self.r.name, color='k', transform=a.transAxes,
           horizontalalignment='right', verticalalignment='top')
    f.tight_layout(pad=0.3) # crop figure to contents
    self.f = f
    return P, freqs


def Plot_PSD(lfp, titlestr):

    colors = ['blue','red', 'black', 'green', 'magenta', 'cyan', 'yellow', 'pink','brown']

    F0, F1 = 0.2, 110 # Hz
    P0, P1 = None, None
    chanis = -1
    width, tres = 10, 5 # sec
    figsize = (3.5, 3.5)
    XSCALE = 'linear'

    sampfreq=25000 #25Khz ec sample rate
    SAMPFREQ=1000 #1Khz LFP sample rate
    NFFT = intround(width * sampfreq)

    NOVERLAP = intround(NFFT - tres * SAMPFREQ)

    ax = plt.subplot(1, 1, 1)
    electrode_list = [0,8] #,2,3,4,5,6,7,8]
    for i in electrode_list:
        print "Plotting ch: ", i

        data = lfp.lfp_traces[i]
        data = filter.notch(data)[0] # remove 60 Hz mains noise, as for SI calc
        # convert data from uV to mV. I think P is in mV^2?:

        P, freqs = mpl.mlab.psd(data/1e3, NFFT=NFFT, Fs=SAMPFREQ, noverlap=NOVERLAP)

        # keep only freqs between F0 and F1:
        f0, f1 = F0, F1 # need to set different local names, since they're not read-only
        if f0 == None:
            f0 = freqs[0]
        if f1 == None:
            f1 = freqs[-1]
        lo, hi = freqs.searchsorted([f0, f1])
        P, freqs = P[lo:hi], freqs[lo:hi]
        # check for and replace zero power values (ostensibly due to gaps in recording)
        # before attempting to convert to dB:
        zis = np.where(P == 0.0) # row and column indices where P has zero power
        if len(zis[0]) > 0: # at least one hit
            P[zis] = np.finfo(np.float64).max # temporarily replace zeros with max float
            minnzval = P.min() # get minimum nonzero value
            P[zis] = minnzval # replace with min nonzero values
        P = 10. * np.log10(P) # convert power to dB wrt 1 mV^2?
        # for better visualization, clip power values to within (P0, P1) dB
        if P0 != None:
            P[P < P0] = P0
        if P1 != None:
            P[P > P1] = P1
        #f = plt.figure(figsize=figsize)
        #a = f.add_subplot(111)
        plt.plot(freqs, P, 'k-',color=colors[i])

    # add SI frequency band limits:
    #a.axvline(x=LFPSILOWBAND[0], c='r', ls='--')
    #a.axvline(x=LFPSILOWBAND[1], c='r', ls='--')
    #a.axvline(x=LFPSIHIGHBAND[0], c='b', ls='--')
    #a.axvline(x=LFPSIHIGHBAND[1], c='b', ls='--')
    ax.axis('tight')
    ax.set_xscale(XSCALE)
    ax.set_ylim(ymin=P[-1]) # use last power value to set ymin
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (dB)")
    #gcfm().window.setWindowTitle(titlestr+' '+XSCALE)
    #ax.tight_layout(pad=0.3) # crop figure to contents

    plt.show()

def Compute_psd_from_tsf(data, SampleFrequency):

    colors = ['blue','red', 'black', 'green', 'magenta', 'cyan', 'yellow', 'pink','brown']

    F0, F1 = 0.2, 110 # Hz
    P0, P1 = None, None
    chanis = -1
    width, tres = 10, 5 # sec
    figsize = (3.5, 3.5)
    XSCALE = 'linear'

    sampfreq=SampleFrequency
    SAMPFREQ=1000 #1Khz LFP sample rate
    NFFT = intround(width * sampfreq)

    NOVERLAP = intround(NFFT - tres * SAMPFREQ)

    #electrode_list = [0,8] #,2,3,4,5,6,7,8]

    #Subsample data to 1Khz
    data = data[::int(SampleFrequency/1E3)]

    #data = filter.notch(data)[0] # remove 60 Hz mains noise, as for SI calc
    # convert data from uV to mV. I think P is in mV^2?:

    P, freqs = mpl.mlab.psd(data/1e3, NFFT=NFFT, Fs=SAMPFREQ, noverlap=NOVERLAP)

    #ax = plt.subplot(111)
    # keep only freqs between F0 and F1:
    f0, f1 = F0, F1 # need to set different local names, since they're not read-only
    if f0 == None:
        f0 = freqs[0]
    if f1 == None:
        f1 = freqs[-1]
    lo, hi = freqs.searchsorted([f0, f1])
    P, freqs = P[lo:hi], freqs[lo:hi]
    # check for and replace zero power values (ostensibly due to gaps in recording)
    # before attempting to convert to dB:
    zis = np.where(P == 0.0) # row and column indices where P has zero power
    if len(zis[0]) > 0: # at least one hit
        P[zis] = np.finfo(np.float64).max # temporarily replace zeros with max float
        minnzval = P.min() # get minimum nonzero value
        P[zis] = minnzval # replace with min nonzero values
    P = 10. * np.log10(P) # convert power to dB wrt 1 mV^2?
    ## for better visualization, clip power values to within (P0, P1) dB
    #if P0 != None:
        #P[P < P0] = P0
    #if P1 != None:
        #P[P > P1] = P1

    return freqs, P
    ##f = plt.figure(figsize=figsize)
    ##a = f.add_subplot(111)
    #plt.plot(freqs, P, 'k-')
    ## add SI frequency band limits:
    ##a.axvline(x=LFPSILOWBAND[0], c='r', ls='--')
    ##a.axvline(x=LFPSILOWBAND[1], c='r', ls='--')
    ##a.axvline(x=LFPSIHIGHBAND[0], c='b', ls='--')
    ##a.axvline(x=LFPSIHIGHBAND[1], c='b', ls='--')
    #ax.axis('tight')
    #ax.set_xscale(XSCALE)
    ##ax.set_ylim(ymin=P[-1]) # use last power value to set ymin
    #ax.set_xlabel("frequency (Hz)")
    #ax.set_ylabel("power (dB)")
    #plt.show()
    #gcfm().window.setWindowTitle(titlestr+' '+XSCALE)
    #ax.tight_layout(pad=0.3) # crop figure to contents

def Plot_PSD_signal(data, sampfreq):

    colors = ['blue','red', 'black', 'green', 'magenta', 'cyan', 'yellow', 'pink','brown']

    F0, F1 = 0.2, 110 # Hz
    P0, P1 = None, None
    chanis = -1
    width, tres = 10, 5 # sec
    figsize = (3.5, 3.5)
    XSCALE = 'linear'

    #sampfreq=25000 #25Khz ec sample rate
    SAMPFREQ=sampfreq #1Khz LFP sample rate
    NFFT = intround(width * sampfreq)

    NOVERLAP = intround(NFFT - tres * SAMPFREQ)

    ax = plt.subplot(1, 1, 1)

    #Enforce 1Khz sampling rate on data
    #data = data[::sampfreq/1000.]

    data = filter.notch(data)[0] # remove 60 Hz mains noise, as for SI calc
    # convert data from uV to mV. I think P is in mV^2?:

    P, freqs = mpl.mlab.psd(data/1e3, NFFT=NFFT, Fs=SAMPFREQ, noverlap=NOVERLAP)

    # keep only freqs between F0 and F1:
    f0, f1 = F0, F1 # need to set different local names, since they're not read-only
    if f0 == None:
        f0 = freqs[0]
    if f1 == None:
        f1 = freqs[-1]
    lo, hi = freqs.searchsorted([f0, f1])
    P, freqs = P[lo:hi], freqs[lo:hi]
    # check for and replace zero power values (ostensibly due to gaps in recording)
    # before attempting to convert to dB:
    zis = np.where(P == 0.0) # row and column indices where P has zero power
    if len(zis[0]) > 0: # at least one hit
        P[zis] = np.finfo(np.float64).max # temporarily replace zeros with max float
        minnzval = P.min() # get minimum nonzero value
        P[zis] = minnzval # replace with min nonzero values
    P = 10. * np.log10(P) # convert power to dB wrt 1 mV^2?
    # for better visualization, clip power values to within (P0, P1) dB
    if P0 != None:
        P[P < P0] = P0
    if P1 != None:
        P[P > P1] = P1
    #f = plt.figure(figsize=figsize)
    #a = f.add_subplot(111)
    plt.plot(freqs, P)

    ax.axis('tight')
    ax.set_xscale(XSCALE)
    #ax.set_ylim(ymin=P[-1]) # use last power value to set ymin
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (dB)")
    #gcfm().window.setWindowTitle(titlestr+' '+XSCALE)
    #ax.tight_layout(pad=0.3) # crop figure to contents

    #*****************************SUBSAMPLE REMOVE ARTIFACTS ***************************
    #Enforce 1Khz sampling rate on data
    
    print len(data)
    sample_step = sampfreq/1000.
    indexes = []
    for i in range(int(len(data)/sample_step)):
        indexes.append(int(i*sample_step))
        
    data = data[indexes]
    print len(data)

    sampfreq = 1000.
    SAMPFREQ=sampfreq #1Khz LFP sample rate
    NFFT = intround(width * sampfreq)

    NOVERLAP = intround(NFFT - tres * SAMPFREQ)

    data = filter.notch(data)[0] # remove 60 Hz mains noise, as for SI calc
    # convert data from uV to mV. I think P is in mV^2?:

    P, freqs = mpl.mlab.psd(data/1e3, NFFT=NFFT, Fs=SAMPFREQ, noverlap=NOVERLAP)

    # keep only freqs between F0 and F1:
    f0, f1 = F0, F1 # need to set different local names, since they're not read-only
    if f0 == None:
        f0 = freqs[0]
    if f1 == None:
        f1 = freqs[-1]
    lo, hi = freqs.searchsorted([f0, f1])
    P, freqs = P[lo:hi], freqs[lo:hi]
    # check for and replace zero power values (ostensibly due to gaps in recording)
    # before attempting to convert to dB:
    zis = np.where(P == 0.0) # row and column indices where P has zero power
    if len(zis[0]) > 0: # at least one hit
        P[zis] = np.finfo(np.float64).max # temporarily replace zeros with max float
        minnzval = P.min() # get minimum nonzero value
        P[zis] = minnzval # replace with min nonzero values
    P = 10. * np.log10(P) # convert power to dB wrt 1 mV^2?
    # for better visualization, clip power values to within (P0, P1) dB
    if P0 != None:
        P[P < P0] = P0
    if P1 != None:
        P[P > P1] = P1
    #f = plt.figure(figsize=figsize)
    #a = f.add_subplot(111)
    plt.plot(freqs, P)

    ax.axis('tight')
    ax.set_xscale(XSCALE)
    #ax.set_ylim(ymin=P[-1]) # use last power value to set ymin
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (dB)")
    #gcfm().window.setWindowTitle(titlestr+' '+XSCALE)
    #ax.tight_layout(pad=0.3) # crop figure to contents

    plt.show()

def Plot_specgram(lfp, show_plot, spec_ch):

    t0=None
    t1=None
    f0=0.1
    f1=110
    p0=-60
    p1=None
    chanis=-1
    width=2
    tres=.25 #Time resolution: bin width for analysis in seconds
    cm=None
    colorbar=False
    title=True
    figsize=(20, 6.5)

    lfp.tres = tres #Save this for later computation

    """Plot a spectrogram from t0 to t1 in sec, from f0 to f1 in Hz, and clip power values
    from p0 to p1 in dB, based on channel index chani of LFP data. chanis=0 uses most
    superficial channel, chanis=-1 uses deepest channel. If len(chanis) > 1, take mean of
    specified chanis. width and tres are in sec. As an alternative to cm.jet (the
    default), cm.gray, cm.hsv cm.terrain, and cm.cubehelix_r colormaps seem to bring out
    the most structure in the spectrogram"""

    F0, F1 = 0.2, 110 # Hz #THESE ARE UNUSED AT THIS TIME;
    P0, P1 = None, None
    chanis = -1
    #width, tres = 10, 5 # sec
    figsize = (3.5, 3.5)
    XSCALE = 'linear'

    sampfreq=lfp.SampleFrequency #1KHZ LFP SAMPLE RATE for Nick's data; Otherwise full sample rates;
    SAMPFREQ=lfp.SampleFrequency #1Khz LFP sample rate

    NFFT = intround(width * sampfreq)

    NOVERLAP = intround(NFFT - tres * SAMPFREQ)

    deepest_ch = spec_ch #np.argmax(np.amax(lfp.lfp_sites,axis=0))

    length = len(lfp.lfp_traces[deepest_ch])

    ts = np.arange(0,len(lfp.lfp_traces[deepest_ch][0:length]),1.0)/sampfreq

    if t0 == None:
        t0, t1 = ts[0], ts[-1] # full duration
    if t1 == None:
        t1 = t0 + 10 # 10 sec window
    if width == None:
        width = uns['LFPWIDTH'] # sec
    if tres == None:
        tres = uns['LFPTRES'] # sec
    assert tres <= width

    NFFT = intround(width * sampfreq)
    noverlap = intround(NFFT - tres * sampfreq)
    #print "noverlap: ", noverlap
    t0i, t1i = ts.searchsorted((t0, t1))
    #ts = ts[t0i:t1i] # constrained set of timestamps, in sec

    if show_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    data = lfp.lfp_traces[deepest_ch][0:length]

    data = filter.notch(data)[0] # remove 60 Hz mains noise

    #np.savez('/home/cat/Downloads/libtfr-master/examples/spec_data.npz', data)
    #print data
    #quit()

    fs = lfp.SampleFrequency 
    lowcut = f0
    highcut = 200

    data = butter_bandpass_filter(data, lowcut, highcut, fs, order = 2)

    #CAT: NB: Martin take mean across all LFP channels here; 
    #if iterable(chanis):
    #    data = data[chanis].mean(axis=0) # take mean of data on chanis
    #else:
    #    data = data[chanis] # get single row of data at chanis

    # convert data from uV to mV, returned t is midpoints of time bins in sec from
    # start of data. I think P is in mV^2?:

    print "INTO specgram"
    P, freqs, t = mpl.mlab.specgram(data/1e3, NFFT=NFFT, Fs=sampfreq, noverlap=noverlap)
    print "OUT of specgram"

    P_Save =P

    #P, freqs, Pt = mpl.mlab.specgram(x, NFFT=NFFT, Fs=self.sampfreq, noverlap=noverlap)
    ## don't convert power to dB, just washes out the signal in the ratio:
    ##P = 10. * np.log10(P)
    #if not relative2t0:
    #    Pt += t0 # convert t to time from start of ADC clock:
    #nfreqs = len(freqs)

    #P, freqs, t, cax = specgram(data/1e3, NFFT=NFFT, Fs=sampfreq, noverlap=noverlap)
    #fig.colorbar(cax)

    # convert t to time from start of acquisition:
    t += t0
    # keep only freqs between f0 and f1:
    if f0 == None:
        f0 = freqs[0]
    if f1 == None:
        f1 = freqs[-1]
    lo, hi = freqs.searchsorted([f0, f1])
    P, freqs = P[lo:hi], freqs[lo:hi]

    # check for and replace zero power values (ostensibly due to gaps in recording)
    # before attempting to convert to dB:
    zis = np.where(P == 0.0) # row and column indices where P has zero power
    if len(zis[0]) > 0: # at least one hit
        P[zis] = np.finfo(np.float64).max # temporarily replace zeros with max float
        minnzval = P.min() # get minimum nonzero value
        P[zis] = minnzval # replace with min nonzero values
    P = 10. * np.log10(P) # convert power to dB wrt 1 mV^2?

    # for better visualization, clip power values to within (p0, p1) dB
    if p0 != None:
        P[P < p0] = p0
    if p1 != None:
        P[P > p1] = p1

    # Label far left, right, top and bottom edges of imshow image. imshow interpolates
    # between these to place the axes ticks. Time limits are
    # set from midpoints of specgram time bins
    extent = t[0], t[-1], freqs[0], freqs[-1]

    #Save specgram data in object 
    lfp.specgram = P
    lfp.extent=extent
    lfp.cm=cm
    
    
    # flip P vertically for compatibility with imshow:
    if show_plot: 
        #print P.shape
        #print P[::-1].shape
        #quit()
        im = ax.imshow(P[::-1], extent=extent, cmap=cm)
        ax.autoscale(enable=True, tight=True)
        ax.axis('tight')
        
        locs, labels = plt.xticks()
        #ax.set_xlim(xmin=0) # acquisition starts at t=0
        
        #plt.xticks(lfp.recording_lengths)
        #plt.xticks(lfp.recording_lengths, lfp.x_labels, fontweight='bold', rotation = 30)

        # turn off annoying "+2.41e3" type offset on x axis:
        #formatter = mpl.ticker.ScalarFormatter(useOffset=False)
        #ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel("time (sec)")
        ax.set_ylabel("frequency (Hz)")
        #titlestr = lastcmd()

    #gcfm().window.setWindowTitle(titlestr)
    titlestr=lfp.header # Sort1.directory + Sort1.name
    if show_plot:
        if title:
            ax.set_title(titlestr)
            #ax.text(0.998, 0.99, '%s' % self.r.name, color='w', transform=a.transAxes,
                   #horizontalalignment='right', verticalalignment='top')
        #f.tight_layout(pad=0.3) # crop figure to contents
        if colorbar:
            f.colorbar(im, pad=0) # creates big whitespace to the right for some reason
        #self.f = f

    #Cat: Compute average power in bin
    if False:
        power=[]
        P=P_Save.T
        print "P Transpose: ", P
        print P.shape
        x = np.arange(0,len(P))*tres
        print "Length P:  ",  len(P)
        for i in range(len(P)):
            power.append(np.sum(P[i][:]))

        power=np.array(power)
        print len(power)
        print "POWER ARRAY: ", power

        ax2=ax.twinx()
        plt.plot(x,power, color='red')
        ax2.set_ylabel('Instantaneous Summed Power (0.1ms bins)')

        #print "Lenght of LFP spectrogram: ", len(x)

        #plt.plot(x,Sort1.inst_firingrate[0:len(x)], color='red', alpha = .5)
        plt.xlim(0,max(x))

    #************************ SECOND PLOT ********************

    #ax = fig.add_subplot(212)
    #titlestr='Kuebler - Instantaneous MUA'
    #ax.set_title(titlestr)

    #plt.plot(x,Sort1.inst_firingrate[0:len(x)], linewidth=.4, color='red', alpha = 1)
    #ax.set_xlabel("time (sec)")
    #ax.set_ylabel("Instataneous Firing rate (Hz; bin=0.1ms)")
    #plt.xlim(0,max(x))
    #mng = plt.get_current_fig_manager()
    #mng.resize(*mng.window.maxsize())

    if show_plot: 
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.savefig(lfp.fname[:-4]+'.png', bbox_inches='tight', dpi=100)

        plt.show()

    return P, freqs, t

def Plot_specgram_tsf(tsf, show_plot, spec_ch):

    t0=None
    t1=None
    f0=0.1
    f1=110
    p0=-60
    p1=None
    chanis=-1
    width=2
    tres=.5 #Time resolution: bin width for analysis in seconds
    cm=None
    colorbar=False
    title=True
    figsize=(20, 6.5)

    tsf.tres = tres #Save this for later computation

    """Plot a spectrogram from t0 to t1 in sec, from f0 to f1 in Hz, and clip power values
    from p0 to p1 in dB, based on channel index chani of LFP data. chanis=0 uses most
    superficial channel, chanis=-1 uses deepest channel. If len(chanis) > 1, take mean of
    specified chanis. width and tres are in sec. As an alternative to cm.jet (the
    default), cm.gray, cm.hsv cm.terrain, and cm.cubehelix_r colormaps seem to bring out
    the most structure in the spectrogram"""


    F0, F1 = 0.2, 110 # Hz #THESE ARE UNUSED AT THIS TIME;
    P0, P1 = None, None
    chanis = -1
    #width, tres = 10, 5 # sec
    figsize = (3.5, 3.5)
    XSCALE = 'linear'

    sampfreq=tsf.SampleFrequency #1KHZ LFP SAMPLE RATE for Nick's data; Otherwise full sample rates;
    SAMPFREQ=tsf.SampleFrequency #1Khz LFP sample rate

    NFFT = intround(width * sampfreq)

    NOVERLAP = intround(NFFT - tres * SAMPFREQ)

    #uns = get_ipython().user_ns
    #self.get_data()
    #ts = self.get_tssec() # full set of timestamps, in sec
    
    #print Sort1.chanpos[lfp.lfp_sites]
    #print lfp.lfp_sites
    #print lfp.lfp_sites_positions

    deepest_ch = spec_ch #np.argmax(np.amax(lfp.lfp_sites,axis=0))
    #print deepest_ch
    
    #length = min(12*lfp.SampleFrequency,len(lfp.lfp_traces[deepest_ch]))
    length = len(tsf.ec_traces[deepest_ch])

    ts = np.arange(0,len(tsf.ec_traces[deepest_ch][0:length]),1.0)/sampfreq


    if t0 == None:
        t0, t1 = ts[0], ts[-1] # full duration
    if t1 == None:
        t1 = t0 + 10 # 10 sec window
    if width == None:
        width = uns['LFPWIDTH'] # sec
    if tres == None:
        tres = uns['LFPTRES'] # sec
    assert tres <= width

    NFFT = intround(width * sampfreq)
    noverlap = intround(NFFT - tres * sampfreq)
    #print "noverlap: ", noverlap
    t0i, t1i = ts.searchsorted((t0, t1))
    #ts = ts[t0i:t1i] # constrained set of timestamps, in sec

    #for i in range(9):
    #print "Plotting ch: ", i

    data = tsf.ec_traces[deepest_ch][0:length]

    #data = filter.notch(data)[0] # remove 60 Hz mains noise

    #np.savez('/home/cat/Downloads/libtfr-master/examples/spec_data.npz', data)
    #print data
    #quit()

    fs = tsf.SampleFrequency 
    lowcut = f0
    highcut = 200

    data = butter_bandpass_filter(data, lowcut, highcut, fs, order = 2)

    #CAT: NB: Martin take mean across all LFP channels here; 
    #if iterable(chanis):
    #    data = data[chanis].mean(axis=0) # take mean of data on chanis
    #else:
    #    data = data[chanis] # get single row of data at chanis

    # convert data from uV to mV, returned t is midpoints of time bins in sec from
    # start of data. I think P is in mV^2?:

    print "INTO specgram"
    P, freqs, t = mpl.mlab.specgram(data/1e3, NFFT=NFFT, Fs=sampfreq, noverlap=noverlap)
    print "OUT of specgram"

    P_Save =P

    #Plot synchrony index
    if False:
        # keep only freqs between f0 and f1, and f2 and f3:
        f0i = 0.1
        f1i = 12.0
        f2i = 12.0
        f3i = 110.0
        
        #f0i, f1i, f2i, f3i = freqs.searchsorted([f0, f1, f2, f3])
        
        lP = P[f0i:f1i] # nsubfreqs x nt
        hP = P[f2i:f3i] # nsubfreqs x nt
        lP = lP.sum(axis=0) # nt
        hP = hP.sum(axis=0) # nt
        
        si = lP/(lP + hP)
        
        x = np.arange(0,len(si),1)
        si = si[::20]
        x = x[::20]

        plt.plot(x, si, color = 'black', linewidth = 3)
        plt.ylabel("Sync Index")
        plt.show()
        quit()

    #P, freqs, t, cax = specgram(data/1e3, NFFT=NFFT, Fs=sampfreq, noverlap=noverlap)
    #fig.colorbar(cax)

    # convert t to time from start of acquisition:
    t += t0
    # keep only freqs between f0 and f1:
    if f0 == None:
        f0 = freqs[0]
    if f1 == None:
        f1 = freqs[-1]
    lo, hi = freqs.searchsorted([f0, f1])
    P, freqs = P[lo:hi], freqs[lo:hi]

    # check for and replace zero power values (ostensibly due to gaps in recording)
    # before attempting to convert to dB:
    zis = np.where(P == 0.0) # row and column indices where P has zero power
    if len(zis[0]) > 0: # at least one hit
        P[zis] = np.finfo(np.float64).max # temporarily replace zeros with max float
        minnzval = P.min() # get minimum nonzero value
        P[zis] = minnzval # replace with min nonzero values
    P = 10. * np.log10(P) # convert power to dB wrt 1 mV^2?

    # for better visualization, clip power values to within (p0, p1) dB
    if p0 != None:
        P[P < p0] = p0
    if p1 != None:
        P[P > p1] = p1

    # Label far left, right, top and bottom edges of imshow image. imshow interpolates
    # between these to place the axes ticks. Time limits are
    # set from midpoints of specgram time bins
    extent = t[0], t[-1], freqs[0], freqs[-1]

    #Save specgram data in object 
    tsf.specgram = P
    
    np.save(tsf.fname[:-4]+'specgram', P)
    
    tsf.extent=extent
    tsf.cm=cm
    
    titlestr=tsf.header # Sort1.directory + Sort1.name
    
    # flip P vertically for compatibility with imshow:
    if show_plot: 
        fig = plt.figure()
        ax = fig.add_subplot(111)  
        
        im = ax.imshow(P[::-1], extent=extent, aspect='auto', cmap=cm)
        ax.autoscale(enable=True, tight=True)
        ax.axis('tight')
        
        locs, labels = plt.xticks()
        #ax.set_xlim(xmin=0) # acquisition starts at t=0


        # turn off annoying "+2.41e3" type offset on x axis:
        #formatter = mpl.ticker.ScalarFormatter(useOffset=False)
        #ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel("time (sec)")
        ax.set_ylabel("frequency (Hz)")
        #titlestr = lastcmd()

        ax.set_title(tsf.tsf_name)
            #ax.text(0.998, 0.99, '%s' % self.r.name, color='w', transform=a.transAxes,
                   #horizontalalignment='right', verticalalignment='top')
        #f.tight_layout(pad=0.3) # crop figure to contents
        if colorbar:
            f.colorbar(im, pad=0) # creates big whitespace to the right for some reason
        #self.f = f

        #mng = plt.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        fig.set_size_inches(30, 10)
        fig.savefig(tsf.fname[:-4]+'.png', bbox_inches='tight', dpi=100)
        #plt.savefig(tsf.fname[:-4]+'.png', bbox_inches='tight', dpi=100)

        plt.show()
        
    #Cat: Compute average power in bin
    if False:
        power=[]
        P=P_Save.T
        print "P Transpose: ", P
        print P.shape
        x = np.arange(0,len(P))*tres
        print "Length P:  ",  len(P)
        for i in range(len(P)):
            power.append(np.sum(P[i][:]))

        power=np.array(power)
        print len(power)
        print "POWER ARRAY: ", power

        ax2=ax.twinx()
        plt.plot(x,power, color='red')
        ax2.set_ylabel('Instantaneous Summed Power (0.1ms bins)')

        #print "Lenght of LFP spectrogram: ", len(x)

        #plt.plot(x,Sort1.inst_firingrate[0:len(x)], color='red', alpha = .5)
        plt.xlim(0,max(x))

    #************************ SECOND PLOT ********************

    #ax = fig.add_subplot(212)
    #titlestr='Kuebler - Instantaneous MUA'
    #ax.set_title(titlestr)

    #plt.plot(x,Sort1.inst_firingrate[0:len(x)], linewidth=.4, color='red', alpha = 1)
    #ax.set_xlabel("time (sec)")
    #ax.set_ylabel("Instataneous Firing rate (Hz; bin=0.1ms)")
    #plt.xlim(0,max(x))
    #mng = plt.get_current_fig_manager()
    #mng.resize(*mng.window.maxsize())


    return P, freqs, t

def Plot_quickspecgram(tsf, show_plot, spec_ch):

    from IPython import get_ipython

    t0=None
    t1=None
    f0=0.1
    f1=110
    p0=-60
    p1=None
    chanis=-1
    width=2
    tres=.5
    cm=None
    colorbar=False
    title=True
    figsize=(20, 6.5)
    
    F0, F1 = 0.2, 110 # Hz #THESE ARE UNUSED AT THIS TIME;
    P0, P1 = None, None
    chanis = -1
    #width, tres = 10, 5 # sec
    figsize = (3.5, 3.5)
    XSCALE = 'linear'

    sampfreq=tsf.SampleFrequency #1KHZ LFP SAMPLE RATE for Nick's data; Otherwise full sample rates;
    SAMPFREQ=tsf.SampleFrequency #1Khz LFP sample rate

    NFFT = intround(width * sampfreq)
    NOVERLAP = intround(NFFT - tres * SAMPFREQ)

    length = len(tsf.ec_traces[spec_ch[0]])

    ts = np.arange(0,len(tsf.ec_traces[spec_ch[0]][0:length]),1.0)/sampfreq

    #print ts

    if t0 == None:
        t0, t1 = ts[0], ts[-1] # full duration
    if t1 == None:
        t1 = t0 + 10 # 10 sec window
    if width == None:
        width = uns['LFPWIDTH'] # sec
    if tres == None:
        tres = uns['LFPTRES'] # sec
    assert tres <= width

    NFFT = intround(width * sampfreq)
    noverlap = intround(NFFT - tres * sampfreq)

    t0i, t1i = ts.searchsorted((t0, t1))

    if show_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    gs = gridspec.GridSpec(4, 4)
    plt.suptitle("Specgram: " + tsf.tsf_name, fontsize = 16)
    
    fs = tsf.SampleFrequency 
    lowcut = f0
    highcut = 200
    
    for specgram_ch in spec_ch:
        data = tsf.ec_traces[specgram_ch][0:length]
        #Band pass filter data
        data = butter_bandpass_filter(data, lowcut, highcut, fs, order = 2)

        #CAT: NB: Martin take mean across all LFP channels here; 
        #if iterable(chanis):
        #    data = data[chanis].mean(axis=0) # take mean of data on chanis
        #else:
        #    data = data[chanis] # get single row of data at chanis
        data = filter.notch(data)[0] # remove 60 Hz mains noise

        print "INTO specgram"
        P, freqs, t = mpl.mlab.specgram(data/1e3, NFFT=NFFT, Fs=sampfreq, noverlap=noverlap)
        print "OUT of specgram"

        P_Save =P

        # convert t to time from start of acquisition:
        t += t0
        # keep only freqs between f0 and f1:
        if f0 == None:
            f0 = freqs[0]
        if f1 == None:
            f1 = freqs[-1]
        lo, hi = freqs.searchsorted([f0, f1])
        P, freqs = P[lo:hi], freqs[lo:hi]

        # check for and replace zero power values (ostensibly due to gaps in recording)
        # before attempting to convert to dB:
        zis = np.where(P == 0.0) # row and column indices where P has zero power
        if len(zis[0]) > 0: # at least one hit
            P[zis] = np.finfo(np.float64).max # temporarily replace zeros with max float
            minnzval = P.min() # get minimum nonzero value
            P[zis] = minnzval # replace with min nonzero values
        P = 10. * np.log10(P) # convert power to dB wrt 1 mV^2?

        # for better visualization, clip power values to within (p0, p1) dB
        if p0 != None:
            P[P < p0] = p0
        if p1 != None:
            P[P > p1] = p1

        extent = t[0], t[-1], freqs[0], freqs[-1]

        # flip P vertically for compatibility with imshow:
        tsf.specgram = P
        tsf.extent=extent
        tsf.cm=cm
        if show_plot: 
            ax = plt.subplot(gs[specgram_ch/4, specgram_ch % 4]) 
            
            im = ax.imshow(P[::-1], extent=extent, cmap=cm)
            ax.autoscale(enable=True, tight=True)
            ax.axis('tight')
            ax.set_xlim(xmin=0) # acquisition starts at t=0

            # turn off annoying "+2.41e3" type offset on x axis:
            formatter = mpl.ticker.ScalarFormatter(useOffset=False)
            ax.xaxis.set_major_formatter(formatter)
            if specgram_ch/4 == 3:
                ax.set_xlabel("time (sec)")
            if specgram_ch % 4 == 0:
                ax.set_ylabel("frequency (Hz)")

            plt.title("Ch: " + str(specgram_ch))
    #mng = plt.get_current_fig_manager()
    #mng.resize(*mng.window.maxsize())
    if show_plot:
        plt.subplots_adjust(left=0.07, right=0.93, top=0.91, bottom=0.06)
        plt.show()

def Compute_specgram_signal(data, SampleFrequency):

    t0=None
    t1=None
    f0=0.1
    f1=110
    p0=-60
    p1=None
    chanis=-1
    width=2
    tres=.5
    cm=None
    colorbar=False
    title=True
    figsize=(20, 6.5)
    
    F0, F1 = 0.2, 110 # Hz #THESE ARE UNUSED AT THIS TIME;
    P0, P1 = None, None
    chanis = -1

    sampfreq=SampleFrequency #1KHZ LFP SAMPLE RATE for Nick's data; Otherwise full sample rates;

    #NFFT = intround(width * sampfreq)
    #NOVERLAP = intround(NFFT - tres * SAMPFREQ)

    length = len(data)

    ts = np.arange(0,len(data),1.0)/sampfreq

    if t0 == None:
        t0, t1 = ts[0], ts[-1] # full duration
    #if t1 == None:
    #    t1 = t0 + 10 # 10 sec window
    if width == None:
        width = uns['LFPWIDTH'] # sec
    if tres == None:
        tres = uns['LFPTRES'] # sec
    assert tres <= width

    NFFT = intround(width * sampfreq)
    noverlap = intround(NFFT - tres * sampfreq)

    t0i, t1i = ts.searchsorted((t0, t1))

    #data = filter.notch(data)[0] # remove 60 Hz mains noise

    print "Computing regular fft specgram"
    P, freqs, t = mpl.mlab.specgram(data/1e3, NFFT=NFFT, Fs=sampfreq, noverlap=noverlap)
    
    # convert t to time from start of acquisition:
    t += t0
    # keep only freqs between f0 and f1:
    if f0 == None:
        f0 = freqs[0]
    if f1 == None:
        f1 = freqs[-1]
    lo, hi = freqs.searchsorted([f0, f1])
    P, freqs = P[lo:hi], freqs[lo:hi]
    #print P
    
    # check for and replace zero power values (ostensibly due to gaps in recording)
    # before attempting to convert to dB:
    zis = np.where(P == 0.0) # row and column indices where P has zero power
    if len(zis[0]) > 0: # at least one hit
        P[zis] = np.finfo(np.float64).max # temporarily replace zeros with max float  #CAT: This can probably be unhacked using nanmax or masked arrays
        minnzval = P.min() # get minimum nonzero value
        P[zis] = minnzval # replace with min nonzero values
    P = 10. * np.log10(P) # convert power to dB wrt 1 mV^2?

    # for better visualization, clip power values to within (p0, p1) dB
    if p0 != None:
        P[P < p0] = p0
    if p1 != None:
        P[P > p1] = p1

    extent = ts[0], ts[-1], freqs[0], freqs[-1]

    return P[::-1], extent

def Compare_sorts(Sort1, Sort2, tsf):

    if (os.path.exists(Sort2.directory+'/comparematch_'+Sort1.name+'_vs_'+Sort2.name+'.csv')==False):
        #Look for best match w/in 10 timesteps; then re-run matcher just for particular cell and unit w. 2-3ms window in case alignemnt is wrong
        dt=1.0*float(tsf.SampleFrequency)*1.0E-3    #time difference in ms allowable between units and cells
                                                    #Generally set at 1ms; 
        max_distance=35.0           #max distance in um allowable between compared units:
        max_ptp_difference = 0.50   #Max size difference allowed between average PTP
        sort = np.zeros((Sort1.n_units, Sort2.n_units), dtype=np.int32)
        sort_flags = []

        for i in range(Sort1.n_units):
            sort_flags.append([0]*Sort1.n_sorted_spikes[i])
        
        #print tsf.Siteloc
        
        Sort1.chanpos=[]
        Sort2.chanpos=[]
        for i in range(len(tsf.Siteloc)/2):
            Sort1.chanpos.append([tsf.Siteloc[i*2],tsf.Siteloc[i*2+1]])
            Sort2.chanpos.append([tsf.Siteloc[i*2],tsf.Siteloc[i*2+1]])
        print Sort1.chanpos
        print Sort2.chanpos
        #quit()

        #Sort2.chanpos = Sort1.chanpos

        sortedspikes_fromcells=[]     #Keeps track of spikes in units that are from other cells 
                                                        #(i.e not from main unit and not noise
        comparematch = [] #Should keep flags '999' if there is no closest match - very rare occassions;
        duplicate = np.zeros(Sort1.n_units, dtype=np.int32)

        for k in range(Sort1.n_units): 
            print ""
            print "**************************************************************"
            print "Checking Cell: ", k, "/", Sort1.n_units, " # spikes: ", Sort1.n_sorted_spikes[k], " ptp: ", Sort1.ptp[k], " uV."
            for j in range(Sort2.n_units):
                #COMPARISON HEURISTICS: distance and PTP

                #Distance between max channels; NB: for .CSV files must compute ch location or request
                unit_distance = (sqrt((Sort1.chanpos[Sort1.maxchan[k]][0]-Sort2.chanpos[Sort2.maxchan[j]][0])**2+
                (Sort1.chanpos[Sort1.maxchan[k]][1]-Sort2.chanpos[Sort2.maxchan[j]][1])**2))
                
                ##Cat more heuristics: can also add (abs(ptp1-ptp2)/max(ptp1,ptp2)<max_ptp_difference):
                #ptp1 = Sort1.ptp[k]
                #ptp2 = Sort2.ptp[j]
                
                if (unit_distance <max_distance): 
                    for p in range(Sort1.n_sorted_spikes[k]):
                        idx=find_nearest(Sort2.units[j], Sort1.units[k][p], dt)
                        if (idx<>None):
                            if (sort_flags[k][p]==0): #Ensure spike hasn't already been detected
                                sort_flags[k][p]=1   #Set flag to 0 to ensure only single detection
                                sort[k][j]+=1
                            else:
                                duplicate[k]+=1
                                #print "****************DUPLICATE SPIKE******************"
                        else:
                            pass
                    print "Vs. Unit: ", j, " dist: ", unit_distance, " # spikes: ", len(Sort2.units[j]), " matches : ", sort[k][j], " ptp: ", Sort2.ptp[j], " uV."
            print sort[k]
            
            if max(sort[k]) > 0:
                #Rerun comparison routine just for highest match; use 2 x size of window in case overlapping units... (Q: in-vitro data no temporal overlap)
                #This is important for spikes that will be temporally overlapping as might be case in silico; 
                
                ind = np.argmax(np.array(sort[k],dtype='int')) #Get index of cell with best match to unit
                print "Rechecking unit: ", k, " size: ", Sort1.n_sorted_spikes[k], " with max match unit: ", ind
                comparematch.append(ind)
                print "1st iteration # spikes match: ", max(sort[k])
                
                sort[k][ind]=0 
                sort_flag2=np.zeros(Sort2.n_sorted_spikes[ind], dtype=int32)  #Set flags back to zero
                for p in range(Sort1.n_sorted_spikes[k]):
                    idx=find_nearest(Sort2.units[ind], Sort1.units[k][p], dt*2)
                    if (idx<>None):
                        if (sort_flag2[idx]==0): #Ensure spike hasn't already been detected
                            sort_flag2[idx]=1   #Set flag to ensure single detection
                            sort[k][ind]+=1
                        #else:
                            #NOT COMPLETELY CORRECT HERE; ON RECHECK CAN FIND ADDITIONAL DUPLICATES... MUST KEEP TRACK
                    else:
                        pass

                print "2nd iteration # spikes match: ", sort[k][ind]
            else:
                print "NO MATCH FOUND ***"
                comparematch.append(999)

        print "Bestmatch array: ", comparematch
        print "Compare array: ", sort

        #Compute Sort2 unit purity:
        purity=[]
        for i in range(len(Sort2.units)):
            #max # spikes matched/Total # spikes in unit; Take truth->sort mapping and transpose it to look at sort-> truth mapping
            purity.append(float(max(sort.T[i])/float(len(Sort2.units[i]))*100)) 
        
        #Compute sortedspikes_fromcells:
        sortedspikes_fromcells=[]
        for i in range(len(Sort2.units)):
            sortedspikes_fromcells.append(sum(sort.T[i])) 
        
        #Compute completeness; search each unit max match and then divide # sorted spikes / # ground truth spikes in cell
        completeness=[]
        for i in range(len(Sort2.units)):
            completeness.append(float(max(sort.T[i]))/float(len(Sort1.units[np.argmax(sort.T[i])])) *100) #max # spikes matched/Total # spikes in closest cell

        np.savetxt(Sort2.directory+'/comparematch_'+Sort1.name+'_vs_'+Sort2.name+'.csv', comparematch, delimiter=",")
        np.savetxt(Sort2.directory+'/comparesort_'+Sort1.name+'_vs_'+Sort2.name+'.csv', sort, delimiter=",")
        np.savetxt(Sort2.directory+'/duplicate_'+Sort1.name+'_vs_'+Sort2.name+'.csv', duplicate, delimiter=",")
        with open(Sort2.directory+'/sort_flags_'+Sort1.name+'_vs_'+Sort2.name+'.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerows(sort_flags)

        np.savetxt(Sort2.directory+'/'+Sort2.name+'_purity.csv', purity, delimiter=",")
        np.savetxt(Sort2.directory+'/'+Sort2.name+'_sortedspikes_fromcells.csv', sortedspikes_fromcells, delimiter=",")
        np.savetxt(Sort2.directory+'/'+Sort2.name+'_completeness.csv', completeness, delimiter=",")
        
        #np.savetxt(sim_dir+'bestmatch.csv', bestmatch, delimiter=",")
        #np.savetxt(sim_dir+'full_sort.csv', sort, delimiter=",")
        #np.savetxt(sim_dir+'duplicate.csv', duplicate, delimiter=",")
        #np.savetxt(sim_dir+'duplicate_mainmatch.csv', duplicate_mainmatch, delimiter=",")
        
    else:
        pass
        #print "Skipping detection (loading previous unit compare data)"


def Match_units(Sort1, Sort2, tsf, tsf2):

    if False:  #have flag that checks of data was already matched and saved

        template=np.zeros((Sort1.n_units,Sort1.nptchans,40), dtype=np.float32)
        max_chan=np.zeros(Sort1.n_units, dtype=np.float32)
        max_ptp=np.zeros(Sort1.n_units, dtype=np.float32)

        ax1 = plt.subplot(1, 1, 1)
    
        if (os.path.isfile(Sort1.directory +'/max_ptp.npz')==True):

            max_ptp=np.load(Sort1.directory+'/max_ptp.npz')
            max_chan=np.load(Sort1.directory+'/max_chan.npz')
            template=np.load(Sort1.directory+'/template.npz')
            template=template[template.keys()[0]]
        else:

            print "COMPUTING DATA"
            for j in range(Sort1.n_units):
                ptp=[[0] for x in range(Sort1.nptchans)]
                for i in range(Sort1.nptchans):  
                    for k in Sort1.units[j]:
                        ptp[i] += tsf.vscale_HP*(max(tsf.ec_traces[i][k-20:k+20])-min(tsf.ec_traces[i][k-20:k+20]))
                        template[j][i] += tsf.vscale_HP*(tsf.ec_traces[i][k-20:k+20])
                        
                max_ptp[j] = max(ptp)/float(len(Sort1.units[j]))
                max_chan[j] = np.argmax(ptp)
                template[j]/=float(len(Sort1.units[j]))
                
                print "Unit: ", j, " max PTP: ", max_ptp[j], " on ch: ", max_chan[j]
                
            np.savez(Sort1.directory+'/max_ptp.npz', max_ptp)
            np.savez(Sort1.directory+'/max_chan.npz', max_chan)
            np.savez(Sort1.directory+'/template.npz', template)
        
        x = np.zeros((tsf.n_electrodes,40),dtype=np.float32)
            
        for j in range(Sort1.n_units):
            for i in range(Sort1.nptchans):
                x[i]= tsf.Siteloc[i*2]/5. + np.array(arange(0,40,1))/10.
                plt.plot(x[i]+j*10, template[j][i]*2-tsf.Siteloc[i*2+1], color='blue', alpha=1.)

            ax1.xaxis.set_major_formatter(plt.NullFormatter())
            plt.xlim(-10,500.)
            plt.ylim(-1600.,200)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.show()


    max_ptp1=np.load(Sort1.directory+'/max_ptp.npz')
    max_chan1=np.load(Sort1.directory+'/max_chan.npz')
    template1=np.load(Sort1.directory+'/template.npz')
    Sort1_templates=template1[template1.keys()[0]]

    max_ptp2=np.load(Sort2.directory+'/max_ptp.npz')
    max_chan2=np.load(Sort2.directory+'/max_chan.npz')
    template2=np.load(Sort2.directory+'/template.npz')
    Sort2_templates=template2[template2.keys()[0]]

    x = np.zeros((tsf.n_electrodes,40),dtype=np.float32)

    for i in range(len(Sort1_templates)):
        #print ""
        print "Matching Sort1 Unit: ", i
        diff_array = Sort2_templates - Sort1_templates[i]
        distances=[]
        #print diff_array[i].shape
        #print len(diff_array)
        for j in range(len(diff_array)):
            #print "Matching Sort2 on Channel: ", j
            #print diff_array[j]
            distances.append(np.sqrt(np.vdot(diff_array[j],diff_array[j])))
        metric= min(distances)
        s2_index=np.array(distances).argmin()

        for ii in range(Sort1.nptchans):
            x[ii]= tsf.Siteloc[ii*2]/5. + np.array(arange(0,40,1))/10.
            plt.plot(x[ii]+i*10, Sort1_templates[i][ii]-tsf.Siteloc[ii*2+1], color='blue', alpha=1.)

            plt.plot(x[ii]+i*10, Sort2_templates[s2_index][ii]-tsf.Siteloc[ii*2+1], color='red', alpha=1.)


        plt.text(10*i+1.5, 25, str(int(metric)), fontsize=10, weight = 'bold', color='black')
        plt.text(10*i, 6, str(len(Sort1.units[i])), fontsize=7,weight = 'bold', color='blue')
        plt.text(10*i+ 4.5, 6, str(len(Sort2.units[s2_index])), weight = 'bold',fontsize=7, color='red')


    #ax1.xaxis.set_major_formatter(plt.NullFormatter())
    plt.xlim(-10,500.)
    plt.ylim(-1600.,200)

    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()

    #plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    #plt.savefig('/home/cat/Desktop/Fig1.eps', format='eps', dpi=1000)
    plt.show()

def Compare_sorts_plot(Sort1, Sort2, tsf):

    comparematch = np.genfromtxt(Sort1.directory+'/comparematch_'+Sort1.name+'_vs_'+Sort2.name++'.csv', dtype='float32', delimiter=",")
    comparesort = np.genfromtxt(Sort1.directory+'/comparesort_'+Sort1.name+'_vs_'+Sort1.name+'.csv', dtype='float32', delimiter=",")
    ptps = np.genfromtxt(Sort1.directory+'/ptps.csv', dtype='float32', delimiter=",")
    size = np.array(Sort1.size) #np.genfromtxt(Sort1.directory+'/size.csv', dtype='float32', delimiter=",")
    size2 = np.array(Sort2.size) #np.genfromtxt(Sort2.directory+'/size.csv', dtype='float32', delimiter=",")

    plt.suptitle('Compare Analysis for: \n"' + Sort1.filename[0:8] + '"  AND  "' + 
    Sort2.filename[0:8] + '" (in vitro)', fontsize = 16)

    scale=1

    #*********************** FIRST PLOT *****************************
    ax = plt.subplot(1, 1, 1)
    title(Sort1.filename[0:8] + ' vs. ' + Sort2.filename[0:8] , fontsize=15)

    #plt.ylabel('% of Total Spikes in Recording', fontsize=17)

    colors = ['blue','red', 'black']
    plt.xlabel('Average PTP (uV)',fontsize=15)
    
    #X-axis; SORT ALL DATA BY PTPS ORDER
    x = ptps
    X = x #Use this array to sort all other data by PTP 
    a = sorted(x)
    tracker = np.arange(0,len(ptps),1)
    tracker = [y for (x,y) in sorted(zip(X,tracker))] #Keeps track of ids sorted by PTP value; use this in place of index

    #Y-axis
    y = np.zeros(len(ptps), dtype=np.float32)
    for i in range(len(ptps)):
        y[i]=i*scale+scale/10. #comparesort[i][comparematch[i]]/size[i]*100

    a=np.array(a)*scale
    #y=a
    plt.scatter(a,y, s=size, alpha=.35, color='blue')

    ss = []
    for i in range(len(ptps)):
        ss.append(size2[int(comparematch[i])])
    plt.scatter(a,y, s=ss, alpha=.35, color='red')

    plt.xlim(0,max(a)+scale/10.)
    plt.ylim(0,max(y)+scale/10.)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


    #***********************************************************************************************************
    #***************************************** SECOND PLOT - ORDERED PLOTS**************************************
    #***********************************************************************************************************

    ax = plt.subplot(1, 2, 1)
    title(Sort1.filename[0:8] + ' vs. ' + Sort2.filename[0:8] , fontsize=15)

    #plt.ylabel('% of Total Spikes in Recording', fontsize=17)

    colors = ['blue','red', 'black']
    plt.xlabel('Average PTP (uV)',fontsize=15)
      
    comparematch = np.genfromtxt(Sort1.directory+'/comparematch_'+Sort2.name+'_vs_'+tsf.tsf_name+'.csv', dtype='float32', delimiter=",")
    comparesort = np.genfromtxt(Sort1.directory+'/comparesort_'+Sort2.name+'_vs_'+tsf.tsf_name+'.csv', dtype='float32', delimiter=",")
    ptps = np.genfromtxt(Sort1.directory+'/ptps.csv', dtype='float32', delimiter=",")
    size = Sort1.size #np.genfromtxt(Sort1.directory+'/size.csv', dtype='float32', delimiter=",")
    size2 = Sort2.size #np.genfromtxt(Sort2.directory+'/size.csv', dtype='float32', delimiter=",")

    title(Sort1.filename[0:8] + ' vs. ' + Sort2.filename[0:8] , fontsize=15)

    #plt.ylabel('% of Total Spikes in Recording', fontsize=17)

    colors = ['blue','red', 'black']
    plt.xlabel('Average PTP (uV)',fontsize=15)
    
    #X-axis; SORT ALL DATA BY PTPS ORDER
    x = ptps
    X = x #Use this array to sort all other data by PTP 
    a = x
    tracker = np.arange(0,len(ptps),1)
    tracker = [y for (x,y) in sorted(zip(X,tracker))]

    size = np.array(size)
    
    #Y-axis
    y = np.zeros(len(ptps), dtype=np.float32)
    for i in range(len(ptps)):
        y[i]=a[i] #tracker.index(i)*scale*10+scale/10.
    #y=a

    a=np.array(a)*scale
    plt.scatter(a,y, s=size/2, alpha=.35, color='blue')
    
    sort1_x=a #Identical to y
    sort1_y=y
    sort1_comparematch=comparematch
    sort1_comparesort=comparesort
    sort1_size = size
    scale2 = max(ptps)

    #************************* SECOND PLOT NO SCALING *****************************
    comparematch = np.genfromtxt(Sort1.directory+'/comparematch_'+Sort2.name+'_vs_'+tsf.tsf_name+'.csv', dtype='float32', delimiter=",")
    comparesort = np.genfromtxt(Sort1.directory+'/comparesort_'+Sort2.name+'_vs_'+tsf.tsf_name+'.csv', dtype='float32', delimiter=",")
    ptps = np.genfromtxt(Sort2.directory+'/ptps.csv', dtype='float32', delimiter=",")
    size = Sort2.size #np.genfromtxt(Sort2.directory+'/size.csv', dtype='float32', delimiter=",")
    size2 = Sort1.size #np.genfromtxt(Sort1.directory+'/size.csv', dtype='float32', delimiter=",")

    plt.xlabel('Average PTP (uV)',fontsize=15)
    plt.ylabel('Average PTP (uV)',fontsize=15)
    #ax.set_yticks([])    
    
    #X-axis; SORT ALL DATA BY PTPS ORDER
    x = ptps
    X = x #Use this array to sort all other data by PTP 
    a = x
    tracker = np.arange(0,len(ptps),1)
    tracker = [y for (x,y) in sorted(zip(X,tracker))]

    size = np.array(size) 
    #Y-axis
    y = np.zeros(len(ptps), dtype=np.float32)
    for i in range(len(ptps)):
        y[i]=a[i] #tracker.index(i)*scale*10+scale/10. 

    a=np.array(a)*scale
    plt.scatter(a+int(math.ceil(scale2/50.0))*50,y, s=size/2, alpha=.35, color='red')

    for i in range(len(sort1_comparesort)): #Loop over all units
        for j in range(len(sort1_comparesort[i])): #Loop over each match in each unit
            if sort1_comparesort[i][j]>0:
                xx = (sort1_x[i], a[j]+int(math.ceil(scale2/50.0))*50)
                yy = (sort1_y[i], y[j])
                alpha_val = (min(sort1_comparesort[i][j]/sort1_size[i]+.05,1.0))
                plt.plot(xx,yy, 'r-', color='black',linewidth=2, alpha=alpha_val)

    #Labels and limits
    plt.xlim(0,int(math.ceil(scale2/50))*50+int(math.ceil(max(ptps)/50.0))*50)
    label_x = np.arange(0,int(math.ceil(scale2/50))*50+int(math.ceil(max(ptps)/50))*50,50)
    my_xticks1 = np.arange(0,int(math.ceil(scale2/50))*50,50).astype('str')
    my_xticks2 = np.arange(0,int(math.ceil(max(ptps)/50))*50, 50).astype('str')
    my_xticks = np.concatenate((my_xticks1,my_xticks2))

    plt.xticks(label_x, my_xticks)
    plt.ylim(bottom=0)

    for i in range(len(label_x)):
        xx=[i*50,i*50]
        yy=[0,10000]
        plt.plot(xx,yy, 'r--', color='black',linewidth=2, alpha=0.5)

    xx=[int(math.ceil(scale2/50))*50,int(math.ceil(scale2/50))*50]
    yy=[0,1000]
    plt.plot(xx,yy,'r--', color='black',linewidth=3)

    #***********************************************************************************************************
    #****************************** NODE Comparisons - PTP BASED ************************************
    #***********************************************************************************************************

    comparematch = np.genfromtxt(Sort1.directory+'/comparematch_vs_'+Sort2.name+'_'+tsf.tsf_name+'.csv', dtype='float32', delimiter=",")
    comparesort = np.genfromtxt(Sort1.directory+'/comparesort_vs_'+Sort2.name+'_'+tsf.tsf_name+'.csv', dtype='float32', delimiter=",")
    ptps = np.genfromtxt(Sort1.directory+'/ptps.csv', dtype='float32', delimiter=",")
    size = Sort1.size #np.genfromtxt(Sort1.directory+'/size.csv', dtype='float32', delimiter=",")
    size2 = Sort2.size #np.genfromtxt(Sort2.directory+'/size.csv', dtype='float32', delimiter=",")

    ax = plt.subplot(1, 2, 2)
    title(Sort1.filename[0:8] + ' vs. ' + Sort2.filename[0:8] , fontsize=15)
    ax.set_yticklabels([])
    
    #plt.ylabel('% of Total Spikes in Recording', fontsize=17)

    colors = ['blue','red', 'black']
    plt.xlabel('Ordered (PTP, no units)',fontsize=15)
    
    #X-axis; SORT ALL DATA BY PTPS ORDER
    x = ptps
    X = x #Use this array to sort all other data by PTP 
    a = x
    tracker = np.arange(0,len(ptps),1)
    tracker = [y for (x,y) in sorted(zip(X,tracker))]

    size = np.array(size)
    
    #Y-axis
    y = np.zeros(len(ptps), dtype=np.float32)
    for i in range(len(ptps)):
        y[i]=tracker.index(i)*scale*50+100
    #y=a

    a=np.array(a)*scale
    plt.scatter(a,y, s=size/2, alpha=.35, color='blue')
    
    sort1_x=a #Identical to y
    sort1_y=y
    sort1_comparematch=comparematch
    sort1_comparesort=comparesort
    sort1_size = size
    scale2 = max(ptps)

    #************************* SWITCH PLOTS *****************************
    comparematch = np.genfromtxt(Sort1.directory+'/comparematch_vs_'+Sort2.name+'_'+tsf.tsf_name+'.csv', dtype='float32', delimiter=",")
    comparesort = np.genfromtxt(Sort1.directory+'/comparesort_vs_'+Sort2.name+'_'+tsf.tsf_name+'.csv', dtype='float32', delimiter=",")
    ptps = np.genfromtxt(Sort2.directory+'/ptps.csv', dtype='float32', delimiter=",")
    size = Sort2.size #np.genfromtxt(Sort2.directory+'/size.csv', dtype='float32', delimiter=",")
    size2 = Sort1.size #np.genfromtxt(Sort1.directory+'/size.csv', dtype='float32', delimiter=",")

    plt.xlabel('Average PTP (uV)',fontsize=15)
    plt.ylabel('Ordered by PTP (no units)',fontsize=15)
    #ax.set_yticks([])    
    
    #X-axis; SORT ALL DATA BY PTPS ORDER
    x = ptps
    X = x #Use this array to sort all other data by PTP 
    a = x
    tracker = np.arange(0,len(ptps),1)
    tracker = [y for (x,y) in sorted(zip(X,tracker))]

    size = np.array(size) 
    #Y-axis
    y = np.zeros(len(ptps), dtype=np.float32)
    for i in range(len(ptps)):
        y[i]=tracker.index(i)*scale*50+100

    a=np.array(a)*scale
    plt.scatter(a+int(math.ceil(scale2/50.0))*50,y, s=size/2, alpha=.35, color='red')

    for i in range(len(sort1_comparesort)): #Loop over all units
        for j in range(len(sort1_comparesort[i])): #Loop over each match in each unit
            if sort1_comparesort[i][j]>0:
                xx = (sort1_x[i], a[j]+int(math.ceil(scale2/50.0))*50)
                yy = (sort1_y[i], y[j])
                alpha_val = (min(sort1_comparesort[i][j]/sort1_size[i]+.05,1.0))
                plt.plot(xx,yy, 'r-', color='black',linewidth=2, alpha=alpha_val)

    #Labels and limits
    plt.xlim(0,int(math.ceil(scale2/50))*50+int(math.ceil(max(ptps)/50))*50)
    label_x = np.arange(0,int(math.ceil(scale2/50))*50+int(math.ceil(max(ptps)/50))*50,50)
    my_xticks1 = np.arange(0,int(math.ceil(scale2/50))*50,50).astype('str')
    my_xticks2 = np.arange(0,int(math.ceil(max(ptps)/50))*50, 50).astype('str')
    my_xticks = np.concatenate((my_xticks1,my_xticks2))

    plt.xticks(label_x, my_xticks)
    plt.ylim(bottom=0)

    for i in range(len(label_x)):
        xx=[i*50,i*50]
        yy=[0,10000]
        plt.plot(xx,yy, 'r--', color='black',linewidth=2, alpha=0.5)

    xx=[int(math.ceil(scale2/50))*50,int(math.ceil(scale2/50))*50]
    yy=[0,1000]
    plt.plot(xx,yy,'r--', color='black',linewidth=3)

    #************************************************

    comparematch = np.genfromtxt(Sort1.directory+'/comparematch_vs_'+Sort2.name+'_'+tsf.tsf_name+'.csv', dtype='float32', delimiter=",")
    comparesort = np.genfromtxt(Sort1.directory+'/comparesort_vs_'+Sort2.name+'_'+tsf.tsf_name+'.csv', dtype='float32', delimiter=",")
    ptps = np.genfromtxt(Sort1.directory+'/ptps.csv', dtype='float32', delimiter=",")
    size = np.array(Sort1.size) #np.genfromtxt(Sort1.directory+'/size.csv', dtype='float32', delimiter=",")
    size2 = np.array(Sort2.size) #np.genfromtxt(Sort2.directory+'/size.csv', dtype='float32', delimiter=",")

    plt.suptitle('Compare Analysis for: \n"' + Sort1.filename[0:8] + '"  AND  "' + 
    Sort2.filename[0:8] + '" : '+tsf.tsf_name, fontsize = 16)

    #VEN-LIKE DIAGRAMS:
    #Need 2 of them still; x-axis = PTP of Sort1, y-axis = % matching of best unit i.e. comparematch.csv file
    #Find radical line and centres of circles to be plotted; 
    #Combine this for up to 3-4 different matching units: i.e. circles should have bubles on them;

    scale=1

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


    #***********************************************************************************************************
    #********************************** NODE Comparisons - PIE CHARTS ******************************************
    #***********************************************************************************************************

    comparematch = np.genfromtxt(Sort1.directory+'/comparematch_vs_'+Sort2.name+'_'+tsf.tsf_name+'.csv', dtype='float32', delimiter=",")
    comparesort = np.genfromtxt(Sort1.directory+'/comparesort_vs_'+Sort2.name+'_'+tsf.tsf_name+'.csv', dtype='float32', delimiter=",")
    ptps = np.genfromtxt(Sort1.directory+'/ptps.csv', dtype='float32', delimiter=",")
    size = Sort1.size #np.genfromtxt(Sort1.directory+'/size.csv', dtype='float32', delimiter=",")
    size2 = Sort2.size #np.genfromtxt(Sort2.directory+'/size.csv', dtype='float32', delimiter=",")

    ax = plt.subplot(1, 1, 1)
    plt.suptitle('Compare Analysis for: \n"' + Sort1.filename[0:8] + '"  &  "' + 
    Sort2.filename[0:8] + '" : '+tsf.tsf_name, fontsize = 16)

    if Sort1.flag == 1:
        title(Sort1.filename[0:8] + ' vs. ' + Sort2.filename[0:8] + ' - Blind Sort ', fontsize=15)
        ax.set_yticklabels([])
    else:
        title('Cell Completeness (Blue) & Unit Purity (Green) ', fontsize=15)
        ax.set_yticklabels([])
    
    colors = ['blue','red', 'black']
    plt.xlabel('Ordered (PTP, no units)',fontsize=15)
    
    #X-axis; SORT ALL DATA BY PTPS ORDER
    x = ptps
    X = x #Use this array to sort all other data by PTP 
    a = x
    tracker = np.arange(0,len(ptps),1)
    tracker = [y for (x,y) in sorted(zip(X,tracker))]

    size = np.array(size)
    
    #Y-axis
    y = np.zeros(len(ptps), dtype=np.float32)
    for i in range(len(ptps)):
        y[i]=tracker.index(i)*50+100

    a=np.array(a)*scale

    #Cat: must always use "truth" in Sort1 - otherwise must make the loop below conditional also (as the one in the next block)
    colors = ['blue','red', 'black']
    if Sort1.flag == 1:
        colors = ['cyan','red', 'black']

    y_scaling=1
    if Sort2.n_units>Sort1.n_units:
        y_scaling = Sort2.n_units/Sort1.n_units
    for i in range(len(a)):
        #Cat: Not enough to find maximum match for each unit/cell;
        #but need to ensure that the matching unit/cell represents the current unit
        #and doesn't have other - larger unit/cells - in it

        itemindex = int(comparematch[i])
        tempint=0
        for j in range(len(comparematch)):
            if tempint < comparesort[j][itemindex]:
                tempint = comparesort[j][itemindex]

        if tempint<=comparesort[i][itemindex]:
            a1=float(comparesort[i][itemindex]/size[i])
        else: 
            a1=0
        print ""
        print a1

        b1=float(sum(comparesort[i])/size[i]-a1)
        c1=1-a1-b1

        #            pie slices,   x loc       y loc   
        draw_pie(ax,[a1, b1, c1], a[i], y[i]*y_scaling, size=size[i]/8, colors=colors)

    #plt.scatter(a,y, s=size/2, alpha=.35, color='blue')

    sort1_x=a #Identical to y
    sort1_y=y
    sort1_comparematch=comparematch
    sort1_comparesort=comparesort
    sort1_size = size
    scale2 = max(ptps)

    #************************* PURITY - RIGHT PLOT *****************************
    comparematch = np.genfromtxt(Sort2.directory+'/comparematch_vs_'+Sort1.name+'_'+tsf.tsf_name+'.csv', dtype='float32', delimiter=",")
    comparesort = np.genfromtxt(Sort2.directory+'/comparesort_vs_'+Sort1.name+'_'+tsf.tsf_name+'.csv', dtype='float32', delimiter=",")
    ptps = np.genfromtxt(Sort2.directory+'/ptps.csv', dtype='float32', delimiter=",")
    size = Sort2.size #np.genfromtxt(Sort1.directory+'/size.csv', dtype='float32', delimiter=",")
    size2 = Sort1.size #np.genfromtxt(Sort2.directory+'/size.csv', dtype='float32', delimiter=",")

    plt.xlabel('Average PTP (uV)',fontsize=15)
    plt.ylabel('Ordered by PTP (no units)',fontsize=15)
    #ax.set_yticks([])    
    
    #X-axis; SORT ALL DATA BY PTPS ORDER
    x = ptps
    X = x #Use this array to sort all other data by PTP 
    a = x
    tracker = np.arange(0,len(ptps),1)
    tracker = [y for (x,y) in sorted(zip(X,tracker))]

    size = np.array(size) 

    #Y-axis
    y = np.zeros(len(ptps), dtype=np.float32)
    for i in range(len(ptps)):
        y[i]=tracker.index(i)*50+100
    #y=a

    a=np.array(a)*scale
    colors = ['green','red', 'black']

    y_scaling=1
    if Sort1.n_units>Sort2.n_units:
        y_scaling = float(Sort1.n_units)/float(Sort2.n_units)

    if Sort1.flag==1: 
        colors = ['cyan','red', 'black'] #Blind COMPARISON; DO NOT SUBTRACT AB

        for i in range(len(a)):
            #Cat: Not enough to find maximum match for each unit/cell;
            #but need to ensure that the matching unit/cell represents the current unit
            #and doesn't have other - larger unit/cells - in it

            itemindex = int(comparematch[i])
            tempint=0
            for j in range(len(comparematch)):
                if tempint < comparesort[j][itemindex]:
                    tempint = comparesort[j][itemindex]

            if tempint<=comparesort[i][itemindex]:
                a1=float(comparesort[i][itemindex]/size[i])
            else: 
                a1=0
            print ""

            b1=float(sum(comparesort[i])/size[i]-a1)
            c1=1-a1-b1

            #            pie slices,   x loc       y loc   
            draw_pie(ax,[a1, b1, c1], a[i]+int(math.ceil(scale2/50.0))*50, y[i], size=size[i]/8, colors=colors)

    else:
        for i in range(len(a)):
            a1=float(max(comparesort[i])/size[i])
            b1=float(sum(comparesort[i])-max(comparesort[i]))/size[i]
            c1=1-a1-b1
            #            pie slices,   x loc       y loc   
            draw_pie(ax,[a1, b1, c1], a[i]+int(math.ceil(scale2/50.0))*50, y[i]*y_scaling, size=size[i]/8, colors=colors)


    #plt.scatter(a+int(math.ceil(scale2/50.0))*50,y, s=size/2, alpha=.35, color='red')

    for i in range(len(sort1_comparesort)): #Loop over all units
        for j in range(len(sort1_comparesort[i])): #Loop over each match in each unit
            if sort1_comparesort[i][j]>0:
                xx = (sort1_x[i], a[j]+int(math.ceil(scale2/50.0))*50)
                yy = (sort1_y[i], y[j]*y_scaling)
                alpha_val = (min(sort1_comparesort[i][j]/sort1_size[i]+.05,1.0))
                plt.plot(xx,yy, 'r-', color='black',linewidth=2, alpha=alpha_val)

    #Labels and limits
    plt.xlim(0,int(math.ceil(scale2/50))*50+int(math.ceil(max(ptps)/50))*50)
    label_x = np.arange(0,int(math.ceil(scale2/50))*50+int(math.ceil(max(ptps)/50))*50,50)
    my_xticks1 = np.arange(0,int(math.ceil(scale2/50))*50,50).astype('str')
    my_xticks2 = np.arange(0,int(math.ceil(max(ptps)/50))*50, 50).astype('str')
    my_xticks = np.concatenate((my_xticks1,my_xticks2))

    plt.xticks(label_x, my_xticks)
    plt.ylim(bottom=0)

    for i in range(len(label_x)):
        xx=[i*50,i*50]
        yy=[0,10000]
        plt.plot(xx,yy, 'r--', color='black',linewidth=2, alpha=0.5)

    xx=[int(math.ceil(scale2/50.0))*50,int(math.ceil(scale2/50.0))*50]
    yy=[0,1000]
    plt.plot(xx,yy,'r--', color='black',linewidth=3)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


    #******************RIGHT PLOT******************************

    comparematch = np.genfromtxt(Sort2.directory+'/comparematch_vs_'+Sort1.name+'_'+tsf.tsf_name+'.csv', dtype='float32', delimiter=",")
    comparesort = np.genfromtxt(Sort2.directory+'/comparesort_vs_'+Sort1.name+'_'+tsf.tsf_name+'.csv', dtype='float32', delimiter=",")
    ptps = np.genfromtxt(Sort2.directory+'/ptps.csv', dtype='float32', delimiter=",")
    size = Sort2.size #np.genfromtxt(Sort1.directory+'/size.csv', dtype='float32', delimiter=",")
    size2 = Sort1.size #np.genfromtxt(Sort2.directory+'/size.csv', dtype='float32', delimiter=",")

    ax = plt.subplot(1, 2, 2)
    if Sort1.flag == 1:
        title(Sort2.filename[0:8] + ' vs. ' + Sort1.filename[0:8] + ' - Blind Sort ', fontsize=15)
        ax.set_yticklabels([])
    else:
        title(Sort2.filename[0:8] + ' vs. ' + Sort1.filename[0:8] + ' - Completeness ', fontsize=15)
        ax.set_yticklabels([])
    
    colors = ['blue','red', 'black']
    plt.xlabel('Ordered (PTP, no units)',fontsize=15)
    
    #X-axis; SORT ALL DATA BY PTPS ORDER
    x = ptps
    X = x #Use this array to sort all other data by PTP 
    a = x
    tracker = np.arange(0,len(ptps),1)
    tracker = [y for (x,y) in sorted(zip(X,tracker))]

    size = np.array(size)
    
    #Y-axis
    y = np.zeros(len(ptps), dtype=np.float32)
    for i in range(len(ptps)):
        y[i]=tracker.index(i)*scale*50+100
    #y=a

    a=np.array(a)*scale
    colors = ['green','red', 'black']

    if Sort1.flag==1: 
        colors = ['cyan','red', 'black'] #Blind COMPARISON; DO NOT SUBTRACT AB

        for i in range(len(a)):
            #Cat: Not enough to find maximum match for each unit/cell;
            #but need to ensure that the matching unit/cell represents the current unit
            #and doesn't have other - larger unit/cells - in it

            itemindex = int(comparematch[i])
            tempint=0
            for j in range(len(comparematch)):
                if tempint < comparesort[j][itemindex]:
                    tempint = comparesort[j][itemindex]

            if tempint<=comparesort[i][itemindex]:
                a1=float(comparesort[i][itemindex]/size[i])
            else: 
                a1=0
            print ""

            b1=float(sum(comparesort[i])/size[i]-a1)
            c1=1-a1-b1

            #            pie slices,   x loc       y loc   
            draw_pie(ax,[a1, b1, c1], a[i], y[i], size=size[i]/8, colors=colors)

    else:
        for i in range(len(a)):
            a1=float(max(comparesort[i])/size[i])
            b1=float(sum(comparesort[i])-max(comparesort[i]))/size[i]
            c1=1-a1-b1
            #            pie slices,   x loc       y loc   
            draw_pie(ax,[a1, b1, c1], a[i], y[i], size=size[i]/8, colors=colors)

    #plt.scatter(a,y, s=size/2, alpha=.35, color='blue')
    
    sort1_x=a #Identical to y
    sort1_y=y
    sort1_comparematch=comparematch
    sort1_comparesort=comparesort
    sort1_size = size
    scale2 = max(ptps)

    #************************* RIGHT PLOT - PURITY *****************************
    comparematch = np.genfromtxt(Sort2.directory+'/comparematch_vs_'+Sort1.name+'_'+tsf.tsf_name+'.csv', dtype='float32', delimiter=",")
    comparesort = np.genfromtxt(Sort2.directory+'/comparesort_vs_'+Sort1.name+'_'+tsf.tsf_name+'.csv', dtype='float32', delimiter=",")
    ptps = np.genfromtxt(Sort1.directory+'/ptps.csv', dtype='float32', delimiter=",")
    size = Sort1.size #np.genfromtxt(Sort2.directory+'/size.csv', dtype='float32', delimiter=",")
    size2 = Sort2.size #np.genfromtxt(Sort1.directory+'/size.csv', dtype='float32', delimiter=",")

    plt.xlabel('Average PTP (uV)',fontsize=15)
    plt.ylabel('Ordered by PTP (no units)',fontsize=15)
    #ax.set_yticks([])    
    
    #X-axis; SORT ALL DATA BY PTPS ORDER
    x = ptps
    X = x #Use this array to sort all other data by PTP 
    a = x
    tracker = np.arange(0,len(ptps),1)
    tracker = [y for (x,y) in sorted(zip(X,tracker))]

    size = np.array(size) 
    #Y-axis
    y = np.zeros(len(ptps), dtype=np.float32)
    for i in range(len(ptps)):
        y[i]=tracker.index(i)*scale*50+100

    a=np.array(a)*scale

    plt.scatter(a+int(math.ceil(scale2/50.0))*50,y, s=size/2, alpha=.35, color='red')

    for i in range(len(sort1_comparesort)): #Loop over all units
        for j in range(len(sort1_comparesort[i])): #Loop over each match in each unit
            if sort1_comparesort[i][j]>0:
                xx = (sort1_x[i], a[j]+int(math.ceil(scale2/50.0))*50)
                yy = (sort1_y[i], y[j])
                alpha_val = (min(sort1_comparesort[i][j]/sort1_size[i]+.05,1.0))
                plt.plot(xx,yy, 'r-', color='black',linewidth=2, alpha=alpha_val)

    #Labels and limits
    plt.xlim(0,int(math.ceil(scale2/50.0))*50+int(math.ceil(max(ptps)/50.0))*50.0)
    label_x = np.arange(0,int(math.ceil(scale2/50.0))*50+int(math.ceil(max(ptps)/50.0))*50,50)
    my_xticks1 = np.arange(0,int(math.ceil(scale2/50.0))*50,50).astype('str')
    my_xticks2 = np.arange(0,int(math.ceil(max(ptps)/50.0))*50, 50).astype('str')
    my_xticks = np.concatenate((my_xticks1,my_xticks2))

    plt.xticks(label_x, my_xticks)
    plt.ylim(bottom=0)

    for i in range(len(label_x)):
        xx=[i*50,i*50]
        yy=[0,10000]
        plt.plot(xx,yy, 'r--', color='black',linewidth=2, alpha=0.5)

    xx=[int(math.ceil(scale2/50.0))*50,int(math.ceil(scale2/50.0))*50]
    yy=[0,1000]
    plt.plot(xx,yy,'r--', color='black',linewidth=3)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


def Check_sort_plot(tsf, Sort1, sim_dir, ptcs_name):

    if False:
        ptps = np.genfromtxt(sim_dir+ptcs_name+'/ptps.csv', dtype='float32', delimiter=",")
        purity = np.genfromtxt(sim_dir+ptcs_name+'/purity.csv', dtype='float32', delimiter=",")
        completeness = np.genfromtxt(sim_dir+ptcs_name+'/completeness.csv', dtype='float32', delimiter=",")
        size = np.genfromtxt(sim_dir+ptcs_name+'/size.csv', dtype='float32', delimiter=",")
        maxchan = np.genfromtxt(sim_dir+ptcs_name+'/maxchan.csv', dtype='float32', delimiter=",")
        sd = np.genfromtxt(sim_dir+ptcs_name+'/sd.csv', dtype='float32', delimiter=",")
        sortedspikes_fromcells = np.genfromtxt(sim_dir+ptcs_name+'/sortedspikes_fromcells.csv', dtype='float32', delimiter=",")
        bestmatch = np.genfromtxt(sim_dir+ptcs_name+'/bestmatch.csv', dtype='float32', delimiter=",")
        cell_ptp = np.genfromtxt(sim_dir+'/cell_ptp.csv', dtype='float32', delimiter=",")
        cell_ptpsd = np.genfromtxt(sim_dir+'/cell_ptpsd.csv', dtype='float32', delimiter=",")
        #cell_ptp = np.genfromtxt(sim_dir+'Traces/'+'/cell_ptp.csv', dtype='float32', delimiter=",")
        #cell_ptpsd = np.genfromtxt(sim_dir+'Traces/'+'/cell_ptpsd.csv', dtype='float32', delimiter=",")
        full_sort = np.genfromtxt(sim_dir+ptcs_name+'/full_sort.csv', dtype='float32', delimiter=",")
        duplicate = np.genfromtxt(sim_dir+ptcs_name+'/duplicate.csv', dtype='float32', delimiter=",")

        #*************** Compute some stats ***********************************
        print "Total ground truth spikes: ", tsf.n_cell_spikes
        n_sorted_spikes=np.array(Sort1.n_sorted_spikes)
        purity=np.array(purity)
        print "Total sorted spikes: ", sum(Sort1.n_sorted_spikes)
        print "Total correct sorted spikes: ", int(sum(Sort1.n_sorted_spikes*purity/100))
        tempreal = sum(Sort1.n_sorted_spikes*purity/100)/sum(Sort1.n_sorted_spikes)*100
        print 'Percent sorted correct: %.2f' % tempreal
        tempreal = sum(purity)/len(purity)
        print 'Unit based percent correct: %.2f' % tempreal

        n_cells = tsf.n_cells
        n_units = Sort1.n_units
        cells = tsf.cell_spikes

        #****************  PLOT ROUTINES **************************************

        plt.suptitle('Sort Analysis for: "' + ptcs_name + '" (In-Vitro Dataset; Rodrigo and Costas)', fontsize = 20)

        #*********************** FIRST PLOT *****************************
        ax = plt.subplot(1, 3, 1)
        title('Cells - Spike Detection Rates' , fontsize=20)

        plt.ylim((0,100))
        plt.xlim((0,250))

        colors = ['blue','red', 'black']

        plt.xlabel('Average PTP (uV)',fontsize=17)
        plt.ylabel('% Completeness\n (Total Spikes Detected From a Cell)',multialignment='center', fontsize=17)

        #****************************** COMPUTE OVERPLIT INDEX ************************************
        a_sum=0
        b_sum=0
        c_sum=0
        oversplitx=[]
        oversplity=[]
        for i in range(n_cells): #Loop over cells
            #print "********************************************************************"
            #print "Searching for cell: ", i
            a = 0.0
            b = 0.0
            c = 1.0
            tempint=0 #cumulative spike count for detected spikes NOT assigned to main unit
            tempint3=0 #Keeps track of # spikes of largest matching unit to cell
            tempint2=1000 #Keeps track of index of max spike unit; 1000 = flag for no match;
            for j in range(n_units): #Loop over units
                if(full_sort[j][i]>0): #Check if any detected spikes in unit from cell 'i'
                    tempint += full_sort[j][i] #Accumulate all sorted spikes belonging to cell 'i'
                    #if full_sort[j][i]>tempint3: #Look for largest unit among all units
                    if bestmatch[j]==i:
                        if full_sort[j][i]>tempint3: #Look for best match unit w. largest # cell spikes
                            tempint2 = j #Keep track of index of max spike unit
                            tempint3=full_sort[j][i] #keep track of # cell spikes in largest, best match unit 

            #print "Total spikes in cell: ", len(cells[i])
            #print "Best matching unit: ", tempint2, " # spikes: ", full_sort[tempint2][i]
            #Cell spikes assigned to best matching unit
            if tempint2<1000:
                a = float(full_sort[tempint2][i])/float(len(cells[i]))
            else:
                a = 0
            a_sum+=a
            
            #tempint-=duplicate[i]

            #print "Other sorted spikes (Overplit Index): ", tempint - full_sort[tempint2][i]
            #Cell spikes detected but assigned to non-best matching unit
            b = float(tempint)/float(len(cells[i]))-a
            b_sum+=b

            #print "Remaining Spikes: ", len(cells[i]) - tempint
            c = 1-a-b
            c_sum+=c

            draw_pie(ax,[a, b, c], cell_ptp[i], float(a)*100, size=125, colors=colors)

            oversplity.append(b)
            oversplitx.append(cell_ptp[i])

        #Compute oversplit sums to show dashed line on graph
        oversplit_sums = np.array([oversplitx, oversplity]).T  #Load oversplit data into 2 columns
        oversplit_sums = oversplit_sums[oversplit_sums[:,0].argsort()]  #Sort data by PTP amplitude (1st column)
        #print "Sorted: ", oversplit_sums

        oversplity_sum=[] #Cumulatively add all oversplit values backwards;
        for i in range(len(oversplity)):
            tempreal=0.0
            for j in range(i,len(oversplity)):
                tempreal+= oversplit_sums[j][1]
            oversplity_sum.append(tempreal)

        oversplity_sum = np.array(oversplity_sum)/float(n_cells)*100
        oversplitx=oversplit_sums[:,0] #Select sorted 1st column of PTP values;
        dash_patch, = ax.plot(oversplitx, oversplity_sum, '--k', color='red', linewidth=5)

        temp_array = [oversplitx, oversplity_sum]
        np.savetxt(sim_dir+ptcs_name+'/oversplit.csv', temp_array, delimiter=",")

        #Plot pie-charts
        blue_patch = mpatches.Patch(color='blue')
        red_patch = mpatches.Patch(color='red')
        black_patch = mpatches.Patch(color='black')

        labels = ['% Correctly assigned', '% Oversplit', '% Missed', '% Oversplit vs PTP ']

        ax.legend([blue_patch, red_patch, black_patch, dash_patch], labels, fontsize=12, loc=0, 
        title="Cell Spikes - Detected")

        #Plot Large pie-chart
        pie_data = a_sum/n_cells, b_sum/n_cells, c_sum/n_cells
        colors = ['blue', 'red', 'black']

        draw_pie(ax,pie_data, 200, 65, size=2500, colors=colors)

        p = ax.axhspan(50.0, 100, facecolor='blue', alpha=0.05)
        p = ax.axvspan(0.0, 200.0, facecolor='1.0', alpha=0.0)

        x = (50,50)
        y = (0,100)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (100,100)
        y = (0,100)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (0,250)
        y = (70,70)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (0,250)
        y = (80,80)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (0,250)
        y = (90,90)
        plt.plot(x,y, 'r--', color='black',linewidth=1)


        #*********************************************** SECOND PLOT *****************************************
        ax = plt.subplot(1, 3, 2)
        title('Units - Contents & Purity' , fontsize=20)
        colors = ['green','red','black','magenta','purple']

        size_scale = 0

        #Plot scatter plots by # of spikes in unit (size)
        if size_scale==0: 
            s = [int(float(x)/5+1) for x in size]
        else:
            #Plot scatter plots by SD of max channel of unit
            s = [int((float(x)-min(sd))*60) for x in sd]

        #ax.scatter(ptps, purity,s=s, alpha=0.4, color='black')

        undersplity=[]
        undersplitx=[]
        noise=[]
        for i in range (len(ptps)):
            a = purity[i]/100.0 #Purity
            b = (sortedspikes_fromcells[i] - n_sorted_spikes[i]*purity[i]/100.0)/ n_sorted_spikes[i] #undersplit
            c = 1-a-b
            draw_pie(ax,[a,b,c], ptps[i], purity[i], size=125, colors=colors)

            undersplity.append(b+c) #*** MUST ADD UNDERSPLIT ERRORS AND NOISE TOGETHER FOR COMPOSITE ERROR METRIC *** 
            undersplitx.append(ptps[i])
            noise.append(c)

        #Compute oversplit sums to show dashed line on graph
        undersplit_sums = np.array([undersplitx, undersplity]).T  #Load oversplit data into 2 columns
        undersplit_sums = undersplit_sums[undersplit_sums[:,0].argsort()]  #Sort data by PTP amplitude (1st column)
        #print "Sorted: ", undersplit_sums

        undersplity_sum=[] #Cumulatively add all oversplit values backwards;
        for i in range(len(undersplity)):
            tempreal=0.0
            for j in range(i,len(undersplity)):
                tempreal+= undersplit_sums[j][1]
            undersplity_sum.append(tempreal)

        undersplity_sum = np.array(undersplity_sum)/float(n_units)*100
        undersplitx=undersplit_sums[:,0] #Select sorted 1st column of PTP values;
        dash_patch, = ax.plot(undersplitx, undersplity_sum, '--k', color='red', linewidth=5)

        temp_array = [undersplitx, undersplity_sum]
        np.savetxt(sim_dir+ptcs_name+'/undersplit.csv', temp_array, delimiter=",")

        green_patch = mpatches.Patch(color='green', label='Single Source Spikes')
        red_patch = mpatches.Patch(color='red', label='Multi Source Spikes')
        black_patch = mpatches.Patch(color='black', label='Noise')

        labels = ['% Correct Spikes', '% Undersplit', '% Noise', '% Undersplit vs. PTP']

        green_patch = mpatches.Patch(color='green', label='% Spikes from Single-Source')
        yellow_patch = mpatches.Patch(color='yellow', label='% Spikes from Multi-Source (Undersplitting)')
        red_patch = mpatches.Patch(color='red', label='Noise')
        ax.legend([green_patch, red_patch, black_patch, dash_patch], labels, loc=4, prop={'size':12}, 
        title='Sorted Spikes - Source Cells')

        #Plot Large pie charts spikes
        a = float(sum(n_sorted_spikes*purity))/float(sum(n_sorted_spikes))/100.
        b = float(sum(sortedspikes_fromcells))/float(sum(n_sorted_spikes))-a
        c = 1-a-b

        pie_data = float(a), float(b),float(c)
        colors = ['green', 'red', 'black']

        draw_pie(ax,pie_data, 200, 65, size=2500, colors=colors)

        p = ax.axhspan(90.0, 100, facecolor='green', alpha=0.05)
        p = ax.axvspan(0.0, 200.0, facecolor='1.0', alpha=0.0)

        plt.ylim((0,100))
        plt.xlim((0,250))
        plt.xlabel('Average PTP (uV)',fontsize=17)
        plt.ylabel('% Purity\n (Spikes in Unit from Unique Cell)',multialignment='center', fontsize=17)

        x = (50,50)
        y = (0,100)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (100,100)
        y = (0,100)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (0,250)
        y = (70,70)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (0,250)
        y = (80,80)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (0,250)
        y = (90,90)
        plt.plot(x,y, 'r--', color='black',linewidth=1)


        #********************************* SIZE PIECHARTS *******************************

        ax = plt.subplot(1, 3, 3)
        plt.ylim((0,100))
        plt.xlim((0,250))

        title('Units - Purity vs. Size of Unit' , fontsize=20)
        colors = ['green','red','black','magenta','purple']

        size_scale = 0

        #Plot scatter plots by # of spikes in unit (size)
        if size_scale==0: 
            s = [int(float(x)/5+1) for x in size]
        else:
            #Plot scatter plots by SD of max channel of unit
            s = [int((float(x)-min(sd))*60) for x in sd]

        #ax.scatter(ptps, purity,s=s, alpha=0.4, color='black')
        size = np.genfromtxt(sim_dir+ptcs_name+'/size.csv', dtype='float32', delimiter=",")

        undersplity=[]
        undersplitx=[]
        noise=[]
        for i in range (len(ptps)):
            a = purity[i]/100.0 #Purity
            b = (sortedspikes_fromcells[i] - n_sorted_spikes[i]*purity[i]/100.0)/ n_sorted_spikes[i] #undersplit
            c = 1-a-b
            #draw_pie(ax,[a,b,c], ptps[i], purity[i], size=size[i]/10., colors=colors)
            plt.scatter(ptps[i],purity[i], s=size[i]/5., color='blue', alpha=0.5)
            undersplity.append(b+c) #*** MUST ADD UNDERSPLIT ERRORS AND NOISE TOGETHER FOR COMPOSITE ERROR METRIC *** 
            undersplitx.append(ptps[i])
            noise.append(c)

        x = (50,50)
        y = (0,100)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (100,100)
        y = (0,100)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (0,250)
        y = (70,70)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (0,250)
        y = (80,80)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (0,250)
        y = (90,90)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

        plt.show()


    #************************************ PURITY VS. COMPLETENESS PLOTS ***********************************************

        plt.suptitle('Sort Analysis for: "' + ptcs_name + '" (In-Vitro Dataset; Rodrigo and Costas)', fontsize = 20)

        #************************************************************* THIRD PLOT *****************************
        ax = plt.subplot(1, 2, 1)
        plt.ylim((0,100))
        plt.xlim((0,250))

        colors = ['cyan','red']

        title('Units - Reliability Metric' , fontsize=20)

        a_sum = 0
        for i in range (len(completeness)):
            a = completeness[i]/100*purity[i]/100.0
            a_sum+= a
            b = 1-a #
            #print b
            draw_pie(ax,[a, b], ptps[i], a*100, size=125)

        blue_patch = mpatches.Patch(color='cyan', label='% Detected spikes')
        white_patch = mpatches.Patch(color='red', label='% Missed spikes')
        #yellow_patch = mpatches.Patch(color='yellow', label='Multi Source Spikes')
        #red_patch = mpatches.Patch(color='red', label='Noise')

        ax.legend([blue_patch, white_patch], ['% Cell Spikes Sorted into Unique Unit', '% Other'], loc=4,
        fontsize=12)

        pie_data = float(a_sum)/len(completeness), 1-float(a_sum)/len(completeness)
        colors = ['cyan', 'red']

        draw_pie(ax,pie_data, 200, 65, size=2500)
        #ax.pie(frac,colors=colors ,labels=labels, autopct='%1.1f%%')

        p = ax.axhspan(50.0, 100, facecolor='cyan', alpha=0.05)
        p = ax.axvspan(0.0, 200.0, facecolor='1.0', alpha=0.0)

        plt.xlabel('Average PTP (uV)',fontsize=17)
        plt.ylabel('% Purity*Completeness Composite\n (Completeness of Cell Activity in Unit)',multialignment='center', fontsize=17)

        x = (50,50)
        y = (0,100)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (100,100)
        y = (0,100)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (0,250)
        y = (70,70)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (0,250)
        y = (80,80)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (0,250)
        y = (90,90)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        ##*********************** FOURTH PLOT ***************************
        ax = plt.subplot(1, 2, 2)

        plt.ylim((0,100))
        plt.xlim((0,100))
        plt.xlabel('% Purity',fontsize=17)
        plt.ylabel('% Completeness',multialignment='center', fontsize=17)

        colors = ['green','blue', 'black']

        a_sum = 0
        print purity
        print completeness

        for i in range (len(completeness)):
            a = purity[i]/200.
            b = completeness[i]/200.
            c = 1 - a - b
            draw_pie(ax,[a, b, c], purity[i], completeness[i], size=125)


        x = (80,80)
        y = (0,100)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (90,90)
        y = (0,100)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (0,250)
        y = (80,80)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        x = (0,250)
        y = (90,90)
        plt.plot(x,y, 'r--', color='black',linewidth=1)

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.show()

#*******************************************************************************************************
#***************************************** MULTISORT PLOTS *****************************
#*******************************************************************************************************

    if False:
        plt.suptitle('MultiSort Comparison (In-Vitro Dataset; Rodrigo and Costas)', fontsize = 20)
        sim_dir = '/media/cat/Data1/in_vitro/all_cell_random_depth/'

        ax = plt.subplot(1, 2, 1)
        title('Oversplit Errors (Cells)' , fontsize=20)

        ax.set_xlim(0,150)
        ax.set_ylim((0,100))
        ax.set_xlabel('Average Peak-to-Peak (PTP) of Spikes in Cell (uV)',fontsize=15)
        ax.set_ylabel('% Oversplit - Cell spikes split across multiple units', fontsize=17)

        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")

        all_cell='james9/'
        oversplit = np.genfromtxt(sim_dir+all_cell+'oversplit.csv', dtype='float32', delimiter=",")
        james_patch, = ax.plot(oversplit[0], oversplit[1], color='green', linewidth=3)

        all_cell='nick/'
        oversplit = np.genfromtxt(sim_dir+all_cell+'oversplit.csv', dtype='float32', delimiter=",")
        nick_patch, = ax.plot(oversplit[0], oversplit[1], color='blue', linewidth=3)

        all_cell='dan/'
        oversplit = np.genfromtxt(sim_dir+all_cell+'oversplit.csv', dtype='float32', delimiter=",")
        dan_patch, = ax.plot(oversplit[0], oversplit[1], color='orange', linewidth=3)

        all_cell='cat/'
        oversplit = np.genfromtxt(sim_dir+all_cell+'oversplit.csv', dtype='float32', delimiter=",")
        cat_patch, = ax.plot(oversplit[0], oversplit[1], color='red', linewidth=3)

        all_cell='dirty_auto/'
        oversplit = np.genfromtxt(sim_dir+all_cell+'oversplit.csv', dtype='float32', delimiter=",")
        dirty_patch, = ax.plot(oversplit[0], oversplit[1], color='cyan', linewidth=3)

        all_cell='clean_auto/'
        oversplit = np.genfromtxt(sim_dir+all_cell+'oversplit.csv', dtype='float32', delimiter=",")
        clean_patch, = ax.plot(oversplit[0], oversplit[1], color='magenta', linewidth=3)


        #************************************ 2ND OVERLAPPING FIGURES ****************************
        ax2 = ax.twinx()

        ax2.set_ylabel('Number of Cells from electrode with spikes > PTP (Normalized)', fontsize=16)
        ax2.set_xlim(0,150)
        ax2.set_ylim(0, 100)
        ax2.set_xlabel('PTP (uV)',fontsize=20)

        plt.yticks([0,43,50,86,100])

        x = np.arange(7.,150.,.01)
        y = (3500/(x-7)+5).tolist()

        dash_patch, = ax2.plot(x,y, '--k', color='black',  linewidth=2)

        x = (150, x[min(range(len(y)), key=lambda i: abs(y[i]-43))])
        a = (43,43)
        ax2.plot(x,a, 'r--', color='black',linewidth=1)

        x = (100,100)
        a = (0,43)
        ax2.plot(x,a, 'r--', color='black', linewidth=1)

        x = np.arange(7.,150.,.01)
        x = (150, x[min(range(len(y)), key=lambda i: abs(y[i]-86))])
        a = (86,86)
        ax2.plot(x,a, 'r--', color='black',  linewidth=1)

        x = (50,50)
        a = (0,86)
        ax2.plot(x,a, 'r--', color='black', linewidth=1)

        #*************** FIRST LEGEND ********************

        labels = ['SS - Nick', 'SS - Auto + Clean', 'SS - Auto Only', 'SS - Catalin',  'Other - James (HHMI)', 'KK - Dan (Allen)']

        leg = plt.legend([nick_patch, clean_patch, dirty_patch, cat_patch, james_patch, dan_patch, dash_patch], labels, 
        loc=1, prop={'size':13}) #

        #*************************************** SECOND PLOT **********************************************

        ax = plt.subplot(1, 2, 2)
        title('Undersplit Errors + Noise (Units)' , fontsize=20)

        ax.set_xlim(0,150)
        ax.set_ylim((0,100))
        ax.set_xlabel('Average Peak-to-Peak (PTP) of Spikes in Unit (uV)',fontsize=15)

        #ax.set_ylabel('% Undersplit - Unit spikes from multiple cells',multialignment='center', fontsize=17)
        #ax.yaxis.tick_left()
        #ax.yaxis.set_label_position("left")

        all_cell='james9/'
        undersplit = np.genfromtxt(sim_dir+all_cell+'undersplit.csv', dtype='float32', delimiter=",")
        dash_patch, = ax.plot(undersplit[0], undersplit[1], color='green', linewidth=3)
        bestmatch = np.genfromtxt(sim_dir+all_cell+'bestmatch.csv', dtype='float32', delimiter=",")
        james_n_cells = float(len(set(bestmatch)))/float(n_cells)*100.0
        #print james_n_cells

        all_cell='nick/'
        undersplit = np.genfromtxt(sim_dir+all_cell+'undersplit.csv', dtype='float32', delimiter=",")
        dash_patch, = ax.plot(undersplit[0], undersplit[1], color='blue', linewidth=3)
        bestmatch = np.genfromtxt(sim_dir+all_cell+'bestmatch.csv', dtype='float32', delimiter=",")
        nick_n_cells = float(len(set(bestmatch)))/n_cells*100.0
        #print nick_n_cells

        all_cell='dan/'
        undersplit = np.genfromtxt(sim_dir+all_cell+'undersplit.csv', dtype='float32', delimiter=",")
        dash_patch, = ax.plot(undersplit[0], undersplit[1], color='orange', linewidth=3)
        bestmatch = np.genfromtxt(sim_dir+all_cell+'bestmatch.csv', dtype='float32', delimiter=",")
        dan_n_cells = float(len(set(bestmatch)))/n_cells*100.0

        all_cell='cat/'
        undersplit = np.genfromtxt(sim_dir+all_cell+'undersplit.csv', dtype='float32', delimiter=",")
        dash_patch, = ax.plot(undersplit[0], undersplit[1],  color='red', linewidth=3)
        bestmatch = np.genfromtxt(sim_dir+all_cell+'bestmatch.csv', dtype='float32', delimiter=",")
        cat_n_cells = float(len(set(bestmatch)))/n_cells*100.0

        all_cell='dirty_auto/'
        undersplit = np.genfromtxt(sim_dir+all_cell+'undersplit.csv', dtype='float32', delimiter=",")
        dash_patch, = ax.plot(undersplit[0], undersplit[1],  color='cyan', linewidth=3)
        bestmatch = np.genfromtxt(sim_dir+all_cell+'bestmatch.csv', dtype='float32', delimiter=",")
        dirty_auto_n_cells = float(len(set(bestmatch)))/n_cells*100.0

        all_cell='clean_auto/'
        undersplit = np.genfromtxt(sim_dir+all_cell+'undersplit.csv', dtype='float32', delimiter=",")
        dash_patch, = ax.plot(undersplit[0], undersplit[1],  color='magenta', linewidth=3)
        bestmatch = np.genfromtxt(sim_dir+all_cell+'bestmatch.csv', dtype='float32', delimiter=",")
        clean_auto_n_cells = float(len(set(bestmatch)))/n_cells*100.0


        #****************** INSERTED PLOT ***********
        ax2 = ax.twinx()

        ax2.set_xlim(0,150)
        ax2.set_ylim(0, 250)

        plt.yticks([0,43,50,86,100])

        x = np.arange(7.,150.,.01)
        y = (3500/(x-7)+5).tolist()

        ax2.plot(x,y, '--k', color='black',  linewidth=3)

        x = (150, x[min(range(len(y)), key=lambda i: abs(y[i]-43))])
        a = (43,43)
        ax2.plot(x,a, 'r--', color='black',linewidth=1)

        x = (100,100)
        a = (0,43)
        ax2.plot(x,a, 'r--', color='black', linewidth=1)

        x = np.arange(7.,150.,.01)
        x = (150, x[min(range(len(y)), key=lambda i: abs(y[i]-86))])
        a = (86,86)
        ax2.plot(x,a, 'r--', color='black',  linewidth=1)

        x = (50,50)
        a = (0,86)
        ax2.plot(x,a, 'r--', color='black', linewidth=1)

        subplots_adjust(hspace = 0.6)
        subplots_adjust(wspace = 0.22)
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.show()

    #**************************************************************************************************************************************
    #******************************************* 4 PLOT COMPARISONS *****************************************************************
    #**************************************************************************************************************************************

def Plot_Composite_metrics(sim_dir, sorted_dirs, sorted_file, anonymous):

    colors=['gray','blue', 'green','cyan', 'red', 'magenta',  'pink', 'yellow']

    #plt.suptitle('Comparison Metrics: ' + sorted_file, weight = 'bold', fontsize = 25)

    #******************* PLOT PURITY METRICS *********************
    ax = plt.subplot(1, 1, 1)

    ax.get_xaxis().set_visible(False)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 100)

    x = (0,200)
    a = (80,80)
    ax.plot(x,a, 'r--', color='black', linewidth=1)

    x = (0,200)
    a = (90,90)
    ax.plot(x,a, 'r--', color='black', linewidth=1)

    font_size = 45
    #print "Total ground truth spikes: ", tsf.n_cell_spikes
    #title('Average Unit Purity & Completeness' , fontsize=40)
    plt.ylabel('Ave. Purity, Completeness and Product (%)', fontsize=font_size)
    
    if sorted_dirs == 'all':
        sorted_files = ['cat_sort/', 'dan_sort/', 'nicksw_sort/', 'nickst_sort/']
    else:
        sorted_files = sorted_dirs
    
    reliability = []
    n_units = []
    for i in range(len(sorted_files)):

        purity = np.genfromtxt(sim_dir+sorted_files[i]+sorted_file+'_purity.csv', dtype='float32', delimiter=",")
        purity=np.array(purity)
        ave_purity=sum(purity)/len(purity)
        
        n_units.append(len(purity))
        
        completeness = np.genfromtxt(sim_dir+sorted_files[i]+sorted_file+'_completeness.csv', dtype='float32', delimiter=",")
        completeness = np.array(completeness)
        ave_completeness = sum(completeness)/len(completeness)

        reliability.append(sum(completeness/100*purity/100.0)/len(purity)*100)
        
        ax.bar(1+i*20, ave_purity, 5, color=colors[i], alpha=0.65)
        ax.bar(6+i*20, ave_completeness, 5, color=colors[i], hatch='//', alpha=0.65)
        ax.bar(11+i*20, reliability[i], 5, color=colors[i], hatch='O', alpha=0.65)

    #*** LEGEND ***
    if True:
        from matplotlib import pyplot

        #LEGEND #1
        #labels = ['Correct', 'Undersplit', 'Noise']
        labels = ['Purity', 'Completeness', 'Pur. * Comp.']
        solid = mpatches.Patch(facecolor = 'white', edgecolor="black")
        hashed = mpatches.Patch(facecolor='white', edgecolor="black", hatch='//')
        black = mpatches.Patch(facecolor='white', edgecolor="black", hatch='O')
        
        first_legend = plt.legend([solid, hashed, black], labels, loc=1, prop={'size':15}) 

        ax = plt.gca().add_artist(first_legend)
        
        #LEGEND #2
        black_patch = mpatches.Patch(color='black')
        blue_patch = mpatches.Patch(color='blue')       #Dan    #2
        red_patch = mpatches.Patch(color='red')         #Sev    #3
        green_patch = mpatches.Patch(color='green')     #Nick   #4
        magenta_patch = mpatches.Patch(color='magenta') #Martin #5
        brown_patch = mpatches.Patch(color='brown')     #Josh   #6
        cyan_patch = mpatches.Patch(color='cyan')       #       #7
        yellow_patch = mpatches.Patch(color='yellow')   #       #8
        pink_patch = mpatches.Patch(color='pink')   #       #8

        if anonymous:
            labels = ['Op #1', 'Op #2', 'Op #3', 'Op #4', 'Op #5', 'Op #6', 'Op #7', 'Op #8'][0:len(sorted_dirs)]
        else:
            labels = sorted_files
        plt.legend([black_patch, blue_patch, green_patch, cyan_patch, red_patch, magenta_patch, pink_patch], labels, fontsize=15, loc=5, 
        title="Operator")

    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(axis='both', which='minor', labelsize=font_size)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()
    #******************** PLOT COMPOSITE METRICS **********************
    ax = plt.subplot(1, 1, 1)

    ax.get_xaxis().set_visible(False)

    ax.set_xlim(0, 200)
    ax.set_ylim(0, 100)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.tick_params(axis='both', which='minor', labelsize=30)
    x = (0,200)
    a = (80,80)
    ax.plot(x,a, 'r--', color='black', linewidth=1)

    x = (0,200)
    a = (90,90)
    ax.plot(x,a, 'r--', color='black', linewidth=1)

    #plt.title('No. of Units and No. of Spikes Sorted', fontsize=20)
    
    ax.set_ylabel('No. of Units Sorted', fontsize=font_size)
    n_units=[]
    n_spikes=[]
    for i in range(len(sorted_files)):

        n_sorted_spikes = np.genfromtxt(sim_dir+sorted_files[i]+sorted_file+'_size.csv', dtype='float32', delimiter=",")
        n_units.append(len(n_sorted_spikes))
        n_spikes.append(sum(n_sorted_spikes))

        ax.bar(1+i*14, n_units[i], 5, color=colors[i], alpha=0.65)

    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(axis='both', which='minor', labelsize=font_size)

    ax2 = ax.twinx()
    ax2.set_ylim(0, 100000)
    ax2.set_xlim(0, 200)
    for i in range(len(sorted_files)):
        ax2.bar(6+i*14, n_spikes[i], 5, color=colors[i],  hatch='//', alpha=0.65)
    ax2.set_ylabel('No. of Spikes Sorted', fontsize=font_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(axis='both', which='minor', labelsize=font_size)
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(1,4))
    #ax2.yaxis.get_major_formatter().set_powerlimits((0,6))
    #ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,1))
    import matplotlib.ticker as mtick
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%1.e'))
    if True:
        from matplotlib import pyplot

        #LEGEND #1
        labels = ['No. Units', 'No. Spikes']
        solid = mpatches.Patch(facecolor = 'pink', edgecolor="black")
        hashed = mpatches.Patch(facecolor="pink", edgecolor="black", hatch='//')
        
        first_legend = plt.legend([solid, hashed], labels, loc=1, prop={'size':13}) 

        ax = plt.gca().add_artist(first_legend)
    
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    #plt.subplots_adjust(left=0.07, right=0.93, top=0.9, bottom=0.1)
    plt.show()

    #******************************************
    #PLot single metric - sorting quality: 

    ax3 = plt.subplot(1, 1, 1)

    ax3.get_xaxis().set_visible(False)
    ax3.set_yticklabels([])

    ax3.set_xlim(0, 200)
    #ax3.set_ylim(0,100)

    #x = (0,200)
    #a = (80,80)
    #ax.plot(x,a, 'r--', color='black', linewidth=1)

    plt.tick_params(axis='both', which='major', labelsize=40)
    plt.tick_params(axis='both', which='minor', labelsize=40)
    #ax.plot(x,a, 'r--', color='black', linewidth=1)

    #plt.title('Purity x Completeness x No. of Spikes Sorted', fontsize=20)
    
    ax3.set_ylabel('Sorting Quality (Relative Units)', fontsize=font_size)
    for i in range(len(sorted_files)):
        print i
        print n_spikes[i]
        print reliability[i]
        temp = reliability[i]  * n_units[i]
        ax3.bar(1+i*14, temp, 10, color=colors[i], alpha=1)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


    quit()

    #****************************************************** SECOND PLOT ***************************************

    ax = plt.subplot(2, 2, 3)
    ax.get_xaxis().set_visible(False)

    ax.set_xlim(0, 200)
    ax.set_ylim(0, 100)
    x = (0,200)
    a = (80,80)
    ax.plot(x,a, 'r--', color='black', linewidth=1)

    x = (0,200)
    a = (90,90)
    ax.plot(x,a, 'r--', color='black', linewidth=1)

    title('% Total (Sum Units): Purity, Undersplit, Noise' , fontsize=15)

    all_cell='nick/'
    purity = np.genfromtxt(sim_dir+all_cell+'purity.csv', dtype='float32', delimiter=",")
    purity=np.array(purity)
    n_sorted_spikes = np.genfromtxt(sim_dir+all_cell+'size.csv', dtype='float32', delimiter=",")
    scale = 1 #sum(n_sorted_spikes)/tsf.n_cell_spikes
    correct = sum(n_sorted_spikes*purity/100)/sum(n_sorted_spikes)*100
    sortedspikes_fromcells = np.genfromtxt(sim_dir+all_cell+'sortedspikes_fromcells.csv', dtype='float32', delimiter=",")
    noise = sum(n_sorted_spikes-sortedspikes_fromcells)/sum(n_sorted_spikes)*100
    other_cells = sum(sortedspikes_fromcells - n_sorted_spikes*purity/100.)/sum(n_sorted_spikes)*100.
    nick_purity = float(sum(n_sorted_spikes*purity))/float(sum(n_sorted_spikes))

    #indent=0
    #plt.bar(4+indent, ave_purity, 6, color='blue')
    #indent+=6
    #plt.bar(4+indent, ave_undersplit, 6, color='blue', hatch='//')
    #indent+=6
    #plt.bar(4+indent, ave_noise, 6, color='black')

    indent=0
    plt.bar(4+indent, correct*scale, 6, color='blue')
    indent+=6
    plt.bar(4+indent, other_cells*scale, 6, color='blue', hatch='//')
    indent+=6
    plt.bar(4+indent, noise*scale, 6, color='black')
    #plt.text(4+indent+4, 102*scale, 'P: %d%%'%int(nick_purity), ha='center', va='bottom')

    all_cell='clean_auto/'
    purity = np.genfromtxt(sim_dir+all_cell+'purity.csv', dtype='float32', delimiter=",")
    purity=np.array(purity)
    n_sorted_spikes = np.genfromtxt(sim_dir+all_cell+'size.csv', dtype='float32', delimiter=",")
    scale = 1 #sum(n_sorted_spikes)/tsf.n_cell_spikes
    correct = sum(n_sorted_spikes*purity/100)/sum(n_sorted_spikes)*100
    sortedspikes_fromcells = np.genfromtxt(sim_dir+all_cell+'sortedspikes_fromcells.csv', dtype='float32', delimiter=",")
    noise = sum(n_sorted_spikes-sortedspikes_fromcells)/sum(n_sorted_spikes)*100
    other_cells = sum(sortedspikes_fromcells - n_sorted_spikes*purity/100.)/sum(n_sorted_spikes)*100.
    cleanauto_purity = float(sum(n_sorted_spikes*purity))/float(sum(n_sorted_spikes))

    indent+=8
    plt.bar(4+indent, correct*scale, 6, color='magenta')
    indent+=6
    plt.bar(4+indent, other_cells*scale, 6, color='magenta', hatch='//')
    indent+=6
    plt.bar(4+indent, noise*scale, 6, color='black')

    all_cell='dirty_auto/'
    purity = np.genfromtxt(sim_dir+all_cell+'purity.csv', dtype='float32', delimiter=",")
    purity=np.array(purity)
    n_sorted_spikes = np.genfromtxt(sim_dir+all_cell+'size.csv', dtype='float32', delimiter=",")
    scale = 1 #sum(n_sorted_spikes)/tsf.n_cell_spikes
    correct = sum(n_sorted_spikes*purity/100)/sum(n_sorted_spikes)*100
    sortedspikes_fromcells = np.genfromtxt(sim_dir+all_cell+'sortedspikes_fromcells.csv', dtype='float32', delimiter=",")
    noise = sum(n_sorted_spikes-sortedspikes_fromcells)/sum(n_sorted_spikes)*100
    other_cells = sum(sortedspikes_fromcells - n_sorted_spikes*purity/100.)/sum(n_sorted_spikes)*100.
    dirtyauto_purity = float(sum(n_sorted_spikes*purity))/float(sum(n_sorted_spikes))

    indent+=8
    plt.bar(4+indent, correct*scale, 6, color='cyan')
    indent+=6
    plt.bar(4+indent, other_cells*scale, 6, color='cyan', hatch='//')
    indent+=6
    plt.bar(4+indent, noise*scale, 6, color='black')

    all_cell='cat/'
    purity = np.genfromtxt(sim_dir+all_cell+'purity.csv', dtype='float32', delimiter=",")
    purity=np.array(purity)
    n_sorted_spikes = np.genfromtxt(sim_dir+all_cell+'size.csv', dtype='float32', delimiter=",")
    scale = 1 #sum(n_sorted_spikes)/tsf.n_cell_spikes
    correct = sum(n_sorted_spikes*purity/100)/sum(n_sorted_spikes)*100
    sortedspikes_fromcells = np.genfromtxt(sim_dir+all_cell+'sortedspikes_fromcells.csv', dtype='float32', delimiter=",")
    noise = sum(n_sorted_spikes-sortedspikes_fromcells)/sum(n_sorted_spikes)*100
    other_cells = sum(sortedspikes_fromcells - n_sorted_spikes*purity/100.)/sum(n_sorted_spikes)*100.
    cat_purity = float(sum(n_sorted_spikes*purity))/float(sum(n_sorted_spikes))

    indent+=8
    plt.bar(4+indent, correct*scale, 6, color='red')
    indent+=6
    plt.bar(4+indent, other_cells*scale, 6, color='red', hatch='//')
    indent+=6
    plt.bar(4+indent, noise*scale, 6, color='black')

    all_cell='james9/'
    purity = np.genfromtxt(sim_dir+all_cell+'purity.csv', dtype='float32', delimiter=",")
    purity=np.array(purity)
    n_sorted_spikes = np.genfromtxt(sim_dir+all_cell+'size.csv', dtype='float32', delimiter=",")
    scale = 1 #sum(n_sorted_spikes)/tsf.n_cell_spikes
    correct = sum(n_sorted_spikes*purity/100)/sum(n_sorted_spikes)*100
    sortedspikes_fromcells = np.genfromtxt(sim_dir+all_cell+'sortedspikes_fromcells.csv', dtype='float32', delimiter=",")
    noise = sum(n_sorted_spikes-sortedspikes_fromcells)/sum(n_sorted_spikes)*100
    other_cells = sum(sortedspikes_fromcells - n_sorted_spikes*purity/100.)/sum(n_sorted_spikes)*100.
    james_purity = float(sum(n_sorted_spikes*purity))/float(sum(n_sorted_spikes))

    indent+=8
    plt.bar(4+indent, correct*scale, 6, color='green')
    indent+=6
    plt.bar(4+indent, other_cells*scale, 6, color='green', hatch='//')
    indent+=6
    plt.bar(4+indent, noise*scale, 6, color='black')

    all_cell='dan/'
    purity = np.genfromtxt(sim_dir+all_cell+'purity.csv', dtype='float32', delimiter=",")
    purity=np.array(purity)
    n_sorted_spikes = np.genfromtxt(sim_dir+all_cell+'size.csv', dtype='float32', delimiter=",")
    scale = 1 #sum(n_sorted_spikes)/tsf.n_cell_spikes
    correct = sum(n_sorted_spikes*purity/100)/sum(n_sorted_spikes)*100
    sortedspikes_fromcells = np.genfromtxt(sim_dir+all_cell+'sortedspikes_fromcells.csv', dtype='float32', delimiter=",")
    noise = sum(n_sorted_spikes-sortedspikes_fromcells)/sum(n_sorted_spikes)*100
    other_cells = sum(sortedspikes_fromcells - n_sorted_spikes*purity/100.)/sum(n_sorted_spikes)*100.
    dan_purity = float(sum(n_sorted_spikes*purity))/float(sum(n_sorted_spikes))

    indent+=8
    plt.bar(4+indent, correct*scale, 6, color='orange')
    indent+=6
    plt.bar(4+indent, other_cells*scale, 6, color='orange', hatch='//')
    indent+=6
    plt.bar(4+indent, noise*scale, 6, color='black')

    #*************** LEGEND ********************

    #labels = ['Solid Color', 'SS - Auto + Clean', 'SS - Auto Only', 'SS - Catalin',  'Other - HHMI', 'KK - Dan']

    labels = ['Correct', 'Undersplit', 'Noise']
    solid = mpatches.Patch(color = 'pink', label = 'Correct')
    hashed = mpatches.Patch(color = 'pink', hatch='//', label = 'Undersplit')
    black = mpatches.Patch(color = 'black', label = 'Noise')

    ax.legend([solid, hashed, black], labels, loc=1, prop={'size':13}) 
    #leg.get_frame().set_alpha(1.0)

    ##**************************** THIRD PLOT ******************************

    ax = plt.subplot(2, 2, 4)
    subplots_adjust(hspace = 0.35)
    #subplots_adjust(wspace = 0.22)
    title('% Correctly Assigned Spikes (Normalized to Total Cell Spikes)' , fontsize=14)

    ax.get_xaxis().set_visible(False)

    # this is another inset axes over the main axes
    #b = axes([0.36, 0.5, .11, .12])
    #setp(b, xlim=(0,135),ylim=(0,100), xticks=[], yticks=[0,25,50,75,100])

    nick_time = 30
    nick_cores = 4
    cat_time = 60
    cat_cores = 1
    autoclean_time = 6
    autoclean_cores = 1
    autodirty_time = 1
    autodirty_cores = 1
    james_time = 200
    james_cores = 16
    dan_time = 85
    dan_cores = 8
    
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 120)

    indent=0
    ax.bar(4, nick_time, 8, color='blue')
    ax.text(8, 1.05*nick_time, '%dm'%int(nick_time), ha='center', va='bottom')

    indent+=16
    ax.bar(24, autoclean_time, 8, color='magenta')
    ax.text(28, 1.05*autoclean_time, '%dm'%int(autoclean_time), ha='center', va='bottom')

    ax.bar(44, autodirty_time, 8, color='cyan')
    ax.text(48, 1.05*autodirty_time, '%dm'%int(autodirty_time), ha='center', va='bottom')

    ax.bar(64, cat_time, 8, color='red')
    ax.text(68, 1.05*cat_time, '%dm'%int(cat_time), ha='center', va='bottom')

    ax.bar(84, james_time, 8, color='green')
    #ax.text(88, 1.05, '???', ha='center', va='bottom')

    ax.bar(104, dan_time, 8, color='orange')
    ax.text(108, 1.05*dan_time, '%dm'%int(dan_time), ha='center', va='bottom')

    title('Sorting Time    &     No. Cores Used', fontsize=15)
    plt.ylabel('Sorting Time (mins)',fontsize=15)


    ax2 = ax.twinx()
    #color=[black]

    #plt.ylabel('# Cores used in sort',color='black', fontsize=15)

    ax2.set_ylim(0, 12)
    ax2.set_xlim(0, 200)

    ax2.bar(12, nick_cores, 8, color='blue', hatch='//')
    ax2.text(16, 1.05*nick_cores, '%dC'%int(nick_cores), ha='center', va='bottom')

    ax2.bar(32, autoclean_cores, 8, color='magenta',hatch='//')
    ax2.text(36, 1.05*autoclean_cores, '%dC'%int(autoclean_cores), ha='center', va='bottom')
    ax2.bar(52, autodirty_cores, 8, color='cyan', hatch='//')
    ax2.text(56, 1.05*autodirty_cores, '%dC'%int(autodirty_cores), ha='center', va='bottom')
    ax2.bar(72, cat_cores, 8, color='red', hatch='//')
    ax2.text(76, 1.05*cat_cores, '%dC'%int(cat_cores), ha='center', va='bottom')

    ax2.bar(92, james_cores, 8, color='green')
    #ax2.text(96, .05, '???', ha='center', va='bottom')
    ax2.bar(112, dan_cores, 8, color='orange', hatch='//')
    ax2.text(116, 1.05*dan_cores, '8', ha='center', va='bottom')

    #ax2.plot(t, s2, 'r.')
    ax2.set_ylabel('# Cores Used in Sorting', color='black')


    #*************** LEGEND ********************

    labels = ['SS - Nick', 'SS - Auto + Clean', 'SS - Auto Only', 'SS - Catalin',  'Sort_9', 'Ephys_Sort']

    leg = plt.legend([nick_patch, clean_patch, dirty_patch, cat_patch, james_patch, dan_patch, dash_patch], labels, 
    loc=4, prop={'size':13}) #

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    #********************PLOT AVERAGE UNIT RELIABILITY
    ax = plt.subplot(2, 2, 2)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 100)

    title('Average Unit Reliability (= Unit Purity * Cell Completeness)' , fontsize=14)

    indent=0

    all_cell='nick/'
    ptp = np.genfromtxt(sim_dir+all_cell+'ptps.csv', dtype='float32', delimiter=",")
    purity = np.genfromtxt(sim_dir+all_cell+'/purity.csv', dtype='float32', delimiter=",")
    purity = np.array(purity)
    completeness = np.genfromtxt(sim_dir+all_cell+'/completeness.csv', dtype='float32', delimiter=",")
    completenss = np.array(completeness)
    a = sum(completeness/100*purity/100.0)/len(purity)*100
    ax.bar(8+indent, a, 8, color='blue')
    a = 0
    counter = 1
    for i in range(len(purity)):
        if ptp[i]>50.0:
            a+=(completeness[i]/100*purity[i]/100.0)
            counter+=1
    a = a/counter*100
    ax.bar(16+indent, a, 8, color='blue',  hatch='//')
    indent+=20

    all_cell='clean_auto/'
    ptp = np.genfromtxt(sim_dir+all_cell+'ptps.csv', dtype='float32', delimiter=",")
    purity = np.genfromtxt(sim_dir+all_cell+'/purity.csv', dtype='float32', delimiter=",")
    purity = np.array(purity)
    completeness = np.genfromtxt(sim_dir+all_cell+'/completeness.csv', dtype='float32', delimiter=",")
    completenss = np.array(completeness)
    a = sum(completeness/100*purity/100.0)/len(purity)*100
    ax.bar(8+indent, a, 8, color='magenta')
    a = 0
    counter = 1
    for i in range(len(purity)):
        if ptp[i]>50.0:
            a+=(completeness[i]/100*purity[i]/100.0)
            counter+=1
    a = a/counter*100
    ax.bar(16+indent, a, 8, color='magenta',  hatch='//')
    indent+=20

    all_cell='dirty_auto/'
    ptp = np.genfromtxt(sim_dir+all_cell+'ptps.csv', dtype='float32', delimiter=",")
    purity = np.genfromtxt(sim_dir+all_cell+'/purity.csv', dtype='float32', delimiter=",")
    purity = np.array(purity)
    completeness = np.genfromtxt(sim_dir+all_cell+'/completeness.csv', dtype='float32', delimiter=",")
    completenss = np.array(completeness)
    a = sum(completeness/100*purity/100.0)/len(purity)*100
    ax.bar(8+indent, a, 8, color='cyan')
    a = 0
    counter = 1
    for i in range(len(purity)):
        if ptp[i]>50.0:
            a+=(completeness[i]/100*purity[i]/100.0)
            counter+=1
    a = a/counter*100
    ax.bar(16+indent, a, 8, color='cyan',  hatch='//')
    indent+=20

    all_cell='cat/'
    ptp = np.genfromtxt(sim_dir+all_cell+'ptps.csv', dtype='float32', delimiter=",")
    purity = np.genfromtxt(sim_dir+all_cell+'/purity.csv', dtype='float32', delimiter=",")
    purity = np.array(purity)
    completeness = np.genfromtxt(sim_dir+all_cell+'/completeness.csv', dtype='float32', delimiter=",")
    completenss = np.array(completeness)
    a = sum(completeness/100*purity/100.0)/len(purity)*100
    ax.bar(8+indent, a, 8, color='red')
    a = 0
    counter = 1
    for i in range(len(purity)):
        if ptp[i]>50.0:
            a+=(completeness[i]/100*purity[i]/100.0)
            counter+=1
    a = a/counter*100
    ax.bar(16+indent, a, 8, color='red',  hatch='//')
    indent+=20

    all_cell='james9/'
    ptp = np.genfromtxt(sim_dir+all_cell+'ptps.csv', dtype='float32', delimiter=",")
    purity = np.genfromtxt(sim_dir+all_cell+'/purity.csv', dtype='float32', delimiter=",")
    purity = np.array(purity)
    completeness = np.genfromtxt(sim_dir+all_cell+'/completeness.csv', dtype='float32', delimiter=",")
    completenss = np.array(completeness)
    a = sum(completeness/100*purity/100.0)/len(purity)*100
    ax.bar(8+indent, a, 8, color='green')
    a = 0
    counter = 1
    for i in range(len(purity)):
        if ptp[i]>50.0:
            a+=(completeness[i]/100*purity[i]/100.0)
            counter+=1
    a = a/counter*100
    ax.bar(16+indent, a, 8, color='green',  hatch='//')
    indent+=20

    all_cell='dan/'
    ptp = np.genfromtxt(sim_dir+all_cell+'ptps.csv', dtype='float32', delimiter=",")
    purity = np.genfromtxt(sim_dir+all_cell+'/purity.csv', dtype='float32', delimiter=",")
    purity = np.array(purity)
    completeness = np.genfromtxt(sim_dir+all_cell+'/completeness.csv', dtype='float32', delimiter=",")
    completenss = np.array(completeness)
    a = sum(completeness/100*purity/100.0)/len(purity)*100
    ax.bar(8+indent, a, 8, color='orange')
    a = 0
    counter = 1
    for i in range(len(purity)):
        if ptp[i]>50.0:
            a+=(completeness[i]/100*purity[i]/100.0)
            counter+=1
    a = a/counter*100
    ax.bar(16+indent, a, 8, color='orange',  hatch='//')

    labels = ['All Units', 'Units > 50uv PTP']
    solid = mpatches.Patch(color = 'pink', label = 'All Units')
    hashed = mpatches.Patch(color = 'pink', hatch='//', label = 'Units > 50uv PTP')

    ax.legend([solid, hashed], labels, loc=1, prop={'size':13}) 

    plt.show()

def Plot_errors(tsf, Sort1):

    print "Plotting sorting errors"

    full_sort = np.genfromtxt(tsf.sim_dir+Sort1.filename+'/full_sort.csv', dtype='float32', delimiter=",")
    bestmatch = np.genfromtxt(tsf.sim_dir+Sort1.filename+'/bestmatch.csv', dtype='int32', delimiter=",")
    ptps = np.genfromtxt(tsf.sim_dir+Sort1.filename+'/ptps.csv', dtype='float32', delimiter=",")

    purity = np.genfromtxt(tsf.sim_dir+Sort1.filename+'/purity.csv', dtype='float32', delimiter=",")
    completeness = np.genfromtxt(tsf.sim_dir+Sort1.filename+'/completeness.csv', dtype='float32', delimiter=",")
    n_sorted_spikes = np.genfromtxt(tsf.sim_dir+Sort1.filename+'/size.csv', dtype='float32', delimiter=",")
    sortedspikes_fromcells = np.genfromtxt(tsf.sim_dir+Sort1.filename+'/sortedspikes_fromcells.csv', dtype='float32', delimiter=",")

    spike_membership=[]
    for i in range(len(bestmatch)):
        spike_membership.append([])
    
    parser = csv.reader(open(tsf.sim_dir+Sort1.filename+'/spike_membership.csv'), delimiter=',')
    counter=0
    for l in parser: 
        for i in range(len(l)):
            spike_membership[counter].append(int(l[i]))
        counter+=1

    x = np.zeros((tsf.n_electrodes,30),dtype=np.float32)
    for i in range(tsf.n_electrodes):
        x[i]= tsf.Siteloc[i*2] + np.array(arange(0,30,1))

    #Plot 6 plots: ground truth, main unit, noise, + additionally mis-identified units
    for k in range(Sort1.n_units):

        #First locate the 2nd and 3rd best matching units - if available
        counts = []
        counts.append(list(count_unique(spike_membership[k])[0]))
        counts.append(list(count_unique(spike_membership[k])[1]))

        print "PTP of Unit: ", ptps[k]
        print counts
        #Remove bestmatch and noise from the 2D list
        for i in range(len(counts[0])):
            if (counts[0][i] == bestmatch[k]):
                del counts[0][i]
                del counts[1][i]
                break
        for i in range(len(counts[0])):
            if (counts[0][i] == 999):
                del counts[0][i]
                del counts[1][i]
                break

        second_match_index=999
        if (len(counts[1])>0):
            max_index, max_value = max(enumerate(counts[1]), key=operator.itemgetter(1))
            if (max_index<>None):
                print "Second unit index: ",  counts[0][max_index]
                second_match_index = counts[0][max_index]
                del counts[0][max_index]
                del counts[1][max_index]

        third_match_index=999
        if (len(counts[1])>0):
            max_index, max_value = max(enumerate(counts[1]), key=operator.itemgetter(1))
            if (max_index<>None):
                print "Third unit index: ",  counts[0][max_index]
                third_match_index =  counts[0][max_index]

        plt.suptitle('Analysis for: "' + Sort1.filename + '" (In-Vitro Data)    Unit: '+str(k+1) + ' / '+str(Sort1.n_units) 
        + ' , No. of spikes: '+str(len(Sort1.units[k])) + ", PTP: " + str(ptps[k]) + 'uV', fontsize = 18)


        # PLOT TEMPLATES
        #**************************************
        ax1 = plt.subplot2grid((2,6),(0,0))
        y1 = np.zeros((tsf.n_electrodes,30), dtype=np.float32)
        counter=0
        for j in range(len(Sort1.units[k])):
            if(spike_membership[k][j]==bestmatch[k]):
                print "Spike: ", j, " of ", len(Sort1.units[k]), " belongs to main unit: ", bestmatch[k]
                counter+=1
                for i in range(tsf.n_electrodes):
                    plt.plot(x[i], tsf.ec_traces[i][Sort1.units[k][j]-15:Sort1.units[k][j]+15]*2-60*tsf.Siteloc[i*2+1], color='blue', alpha=0.2)
                    y1[i]+=tsf.ec_traces[i][Sort1.units[k][j]-15:Sort1.units[k][j]+15]*2
                    #print y1[i]
                    #time.sleep(10)

        y1=y1/float(counter)
        for i in range(tsf.n_electrodes):
            plt.plot(x[i],y1[i]-60*tsf.Siteloc[i*2+1], color='black',linewidth=3,alpha=1.0)

        title('Best Match Cell: ' +str(bestmatch[k]) + '\n No. Spikes: '+str(counter)+" / "+str(len(Sort1.units[k])) , fontsize=13)
        main_spikes = counter
        #Cat: Plot legends; the scale has to be multiplied by V_HP_Scale
        #ax1.text(-25*1E+4/tsf.SampleFrequency,-12500, '100uV', rotation=90)
        #ax1.text(-19*1E+4/tsf.SampleFrequency,-14650, '1.0ms', rotation=0)
        #x_local=[-18.0,-18.0]
        #y_local=[-12000.0,-12000.0-100*2*10]
        #ax1.plot(x_local,y_local, linewidth=5, color='black') 

        #x_local=[-18.0,-8.0]
        #y_local=[-14000.0,-14000]
        #ax1.plot(x_local,y_local, linewidth=5, color='black')

        ax1.set_ylim(-15000,1000)
        ax1.xaxis.set_major_formatter(plt.NullFormatter())
        ax1.yaxis.set_major_formatter(plt.NullFormatter())


        #**************************************
        ax4 = plt.subplot2grid((2,6),(0,1))
        y3 = np.zeros((tsf.n_electrodes,30), dtype=np.float32)
        counter=0
        for j in range(len(Sort1.units[k])):
            if(spike_membership[k][j]==second_match_index and second_match_index<>999):
                counter+=1
                print "Spike: ", j, " of ", len(Sort1.units[k]), " belongs to second unit"
                for i in range(tsf.n_electrodes):
                    plt.plot(x[i], tsf.ec_traces[i][Sort1.units[k][j]-15:Sort1.units[k][j]+15]*2-60*tsf.Siteloc[i*2+1], color='magenta', alpha=0.2)
                    y3[i]+=tsf.ec_traces[i][Sort1.units[k][j]-15:Sort1.units[k][j]+15]*2

        if(second_match_index==999):
            title('No additional units', fontsize=15)
        else:
            title('2nd Best Match cell: '+  str(second_match_index)+ '\nNo. of Spikes: '+str(counter)+ ' / '+str(len(Sort1.units[k])), fontsize=13)
            y3=y3/float(counter)
            for i in range(tsf.n_electrodes):
                plt.plot(x[i],y3[i]-60*tsf.Siteloc[i*2+1], color='black',linewidth=3,alpha=1.0)

        second_spikes = counter
        plt.ylim(-15000,1000)
        ax4.xaxis.set_major_formatter(plt.NullFormatter())
        ax4.yaxis.set_major_formatter(plt.NullFormatter())

        #**************************************
        ax5 = plt.subplot2grid((2,6),(0,2))
        y4 = np.zeros((tsf.n_electrodes,30), dtype=np.float32)
        counter=0
        for j in range(len(Sort1.units[k])):
            if(spike_membership[k][j]==third_match_index and third_match_index<>999):
                counter+=1
                print "Spike: ", j, " of ", len(Sort1.units[k]), " belongs to third unit"
                for i in range(tsf.n_electrodes):
                    plt.plot(x[i], tsf.ec_traces[i][Sort1.units[k][j]-15:Sort1.units[k][j]+15]*2-60*tsf.Siteloc[i*2+1], color='green', alpha=0.2)
                    y4[i]+=tsf.ec_traces[i][Sort1.units[k][j]-15:Sort1.units[k][j]+15]*2

        if(third_match_index==999):
            title('No additional units', fontsize=15)
        else:
            title('3rd Best Match cell: '+  str(third_match_index)+ '\nNo. of Spikes: '+str(counter)+ ' / '+str(len(Sort1.units[k])), fontsize=13)

            y4=y4/float(counter)
            for i in range(tsf.n_electrodes):
                plt.plot(x[i],y4[i]-60*tsf.Siteloc[i*2+1], color='black',linewidth=3,alpha=1.0)

        third_spikes = counter

        plt.ylim(-15000,1000)
        ax5.xaxis.set_major_formatter(plt.NullFormatter())
        ax5.yaxis.set_major_formatter(plt.NullFormatter())

        #**************************************
        ax2 = plt.subplot2grid((2,6),(1,0))
        title('Matching Cell: ' +str(bestmatch[k])+ "\n Total Spikes: "+str(len(tsf.cell_spikes[bestmatch[k]])), fontsize=13)
        y0 = np.zeros((tsf.n_electrodes,30), dtype=np.float32)
        counter=0
        for j in range(len(tsf.cell_spikes[bestmatch[k]])):
            print "Cell: ", bestmatch[k], " Spike: ", j, " of ", len(tsf.cell_spikes[bestmatch[k]])
            counter+=1
            if(counter<1000):
                for i in range(tsf.n_electrodes):
                    plt.plot(x[i], tsf.ec_traces[i][tsf.cell_spikes[bestmatch[k]][j]-15:tsf.cell_spikes[bestmatch[k]][j]+15]*2-60*tsf.Siteloc[i*2+1], 
                    color='black', alpha=0.2)
                    y0[i]+=tsf.ec_traces[i][tsf.cell_spikes[bestmatch[k]][j]-15:tsf.cell_spikes[bestmatch[k]][j]+15]*2

        y0=y0/float(min(1000,counter))
        for i in range(tsf.n_electrodes):
            plt.plot(x[i],y0[i]-60*tsf.Siteloc[i*2+1], linewidth=3,color='white')

        plt.ylim(-15000,1000)
        ax2.xaxis.set_major_formatter(plt.NullFormatter())
        ax2.yaxis.set_major_formatter(plt.NullFormatter())

        #**************************************
        ax3 = plt.subplot2grid((2,6),(1,1))
        y2 = np.zeros((tsf.n_electrodes,30), dtype=np.float32)
        counter=0
        for j in range(len(Sort1.units[k])):
            if(spike_membership[k][j]==999):
                print "Spike: ", j, " of ", len(Sort1.units[k]), " is noise."
                counter+=1
                if(counter<1000): #Plot only first 1000 noise spikes;
                    for i in range(tsf.n_electrodes):
                        if Sort1.units[k][j]>15: #Skip spikes that are in the first 15 timesteps in the record;
                            plt.plot(x[i], tsf.ec_traces[i][Sort1.units[k][j]-15:Sort1.units[k][j]+15]*2-60*tsf.Siteloc[i*2+1], color='red', alpha=0.2)
                            y2[i]+=tsf.ec_traces[i][Sort1.units[k][j]-15:Sort1.units[k][j]+15]*2

        y2=y2/float(min(1000,counter))
        for i in range(tsf.n_electrodes):
            plt.plot(x[i],y2[i]-60*tsf.Siteloc[i*2+1], linewidth=3,color='black')
        noise_spikes=counter

        plt.ylim(-15000,1000)
        title('Noise (or In-Vivo Unit) \n No of Spikes: ' + str(counter)+" / "+str(len(Sort1.units[k])), fontsize=13)
        ax3.xaxis.set_major_formatter(plt.NullFormatter())
        ax3.yaxis.set_major_formatter(plt.NullFormatter())



        #Plot purity and completnes pie_charts
        #**************************************
        ax6 = plt.subplot2grid((2,6),(1,2))

        ##PURITY
        #a = purity[k]/100.0 
        #b = (sortedspikes_fromcells[k] - n_sorted_spikes[k]*purity[k]/100.0)/n_sorted_spikes[k] #undersplit
        #c = 1-a-b
        #colors = ['green','red', 'black']
        #draw_pie(ax6,[a, b, c], 500,3000 , size=2000, colors=colors)

        ##COMPLETENESS
        #colors = ['blue', 'black']
        #a = completeness[k]/100.0
        #b = 1-a
        #draw_pie(ax6,[a,b], 500,1000, size=2000, colors=colors)

        #title('Purity & Completeness', fontsize=13)
        title('Unit Composition (%)', fontsize=13)
        #colors = ['blue', 'magenta', 'green', 'red']
        a = main_spikes/ float(len(Sort1.units[k]))
        b = second_spikes/float(len(Sort1.units[k]))
        c = third_spikes/float(len(Sort1.units[k]))
        e = noise_spikes/float(len(Sort1.units[k]))
        d = 1-a-b-c-e
        #draw_pie(ax6,[a,b,c,d], 500,1000, size=2000, colors=colors)

        ax6.bar(25, 100, 50, color='red')
        ax6.bar(25, (a+b+c+d)*100, 50, color='pink')
        ax6.bar(25, (a+b+c)*100, 50, color='green')
        ax6.bar(25, (a+b)*100, 50, color='magenta')
        ax6.bar(25, a*100, 50, color='blue')

        plt.ylim(0,100)
        plt.xlim(0,100)
        ax6.xaxis.set_major_formatter(plt.NullFormatter())
        #ax6.yaxis.set_major_formatter(plt.NullFormatter())

        #**************************************
        ax7 = plt.subplot2grid((2,6),(0,3), rowspan=2)
        for i in range(tsf.n_electrodes):
            plt.plot(x[i],y1[i]-60*tsf.Siteloc[i*2+1], color='blue',linewidth=3,alpha=0.6)
            plt.plot(x[i],y2[i]-60*tsf.Siteloc[i*2+1], color='red',linewidth=3,alpha=1)

        title('Correct vs. Noise', fontsize=15)
        plt.ylim(-15000,1000)
        ax7.xaxis.set_major_formatter(plt.NullFormatter())
        ax7.yaxis.set_major_formatter(plt.NullFormatter())

        #ax7.text(-25,-12500, '100uV', rotation=90)
        #ax7.plot([-18.0,-18.0],[-12000.0,-12000.0-100*2*10], linewidth=5, color='black') #Cat: the scale has to be multiplied by V_HP_Scale

        #ax7.text(-19,-14650, '1.0ms', rotation=0)
        #ax7.plot([-18.0,-8.0],[-14000.0,-14000.0], linewidth=5, color='black')

        #**************************************
        ax8 = plt.subplot2grid((2,6),(0,4), rowspan=2)
        if(second_match_index<>999):
            for i in range(tsf.n_electrodes):
                plt.plot(x[i],y1[i]-60*tsf.Siteloc[i*2+1], color='blue',linewidth=3,alpha=.6)
                plt.plot(x[i],y3[i]-60*tsf.Siteloc[i*2+1], color='magenta',linewidth=3,alpha=1)
            title('Correct vs. 2nd Largest', fontsize=15)
        else:
            title('No additional matches')
        plt.ylim(-15000,1000)
        ax8.xaxis.set_major_formatter(plt.NullFormatter())
        ax8.yaxis.set_major_formatter(plt.NullFormatter())

        #**************************************
        ax9 = plt.subplot2grid((2,6),(0,5), rowspan=2)
        if(third_match_index<>999):
            for i in range(tsf.n_electrodes):
                plt.plot(x[i],y1[i]-60*tsf.Siteloc[i*2+1], color='blue',linewidth=3,alpha=.6)
                plt.plot(x[i],y4[i]-60*tsf.Siteloc[i*2+1], color='green',linewidth=3, alpha=1)
            title('Correct vs. 3rd Largest', fontsize=15)
        else:
            title('No additional matches')

        plt.ylim(-15000,1000)
        ax9.xaxis.set_major_formatter(plt.NullFormatter())
        ax9.yaxis.set_major_formatter(plt.NullFormatter())

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

        #pylab.savefig(tsf.sim_dir+Sort1.filename+'/Unit_'+str(k)+'.png')
        #plt.get_current_fig_manager().window.showMaximized()
        #plt.savefig(tsf.sim_dir+Sort1.filename+'/Unit_'+str(k)+'.png', bbox_inches='tight', dpi=600)
        plt.show()

def Make_tsf(tsf, Sort1, file_out, unit_no, index_1, index_2):
    "Making TSF: use only iformat = 1002"

    iformat=1002
    f = open(file_out, 'wb')
    f.write(tsf.header)
    f.write(struct.pack('i', iformat))
    f.write(struct.pack('i', tsf.SampleFrequency))
    f.write(struct.pack('i', tsf.n_electrodes))
    f.write(struct.pack('i', tsf.n_vd_samples))
    f.write(struct.pack('f', tsf.vscale_HP))

    for i in range (tsf.n_electrodes):
        f.write(struct.pack('h', tsf.Siteloc[i*2]))
        f.write(struct.pack('h', tsf.Siteloc[i*2+1]))
        f.write(struct.pack('i', tsf.Readloc[i]))

    print "Writing data"
    for i in range (tsf.n_electrodes):
        print "Writing ch: ", i
        tsf.ec_traces[i].tofile(f)

    #Determine spike sets to save as ground truth
    maxchan = np.genfromtxt(tsf.sim_dir+Sort1.filename+'/maxchan.csv', dtype='float32', delimiter=",")

    spike_membership=[]
    for i in range(len(maxchan)):
        spike_membership.append([])
    
    parser = csv.reader(open(tsf.sim_dir+Sort1.filename+'/spike_membership.csv'), delimiter=',')
    counter=0
    for l in parser: 
        for i in range(len(l)):
            spike_membership[counter].append(int(l[i]))
        counter+=1

    print spike_membership[unit_no]

    fake_spike_times=[]
    fake_spike_assignment=[]
    fake_spike_channels=[]
    n_spikes=0
    for i in range(len(spike_membership[unit_no])):
        if(spike_membership[unit_no][i]==index_1):
            fake_spike_times.append(Sort1.units[unit_no][i])
            fake_spike_assignment.append(1)
            fake_spike_channels.append(maxchan[unit_no+1])
            n_spikes+=1
        if(spike_membership[unit_no][i]==index_2):
            fake_spike_times.append(Sort1.units[unit_no][i])
            fake_spike_assignment.append(2)
            fake_spike_channels.append(maxchan[unit_no+1])
            n_spikes+=1
    fake_spike_times=np.array(fake_spike_times, dtype=np.int32)
    fake_spike_assignment=np.array(fake_spike_assignment, dtype=np.int32)
    fake_spike_channels=np.array(fake_spike_channels, dtype=np.int32)

    print fake_spike_times[0:10]
    print fake_spike_assignment[0:10]
    print fake_spike_channels[0:10]


    f.write(struct.pack('i', n_spikes)) #Write # of fake spikes

    print "No. cell spikes: ", n_spikes
    if (n_spikes>0):
        if (iformat==1001):
            f.write(struct.pack('i', 30)) # = struct.unpack('i',fin.read(4))[0] 
            f.write(struct.pack('i', n_spikes)) #Write # of fake spikes
        fake_spike_times.tofile(f)
        fake_spike_assignment.tofile(f) 
        fake_spike_channels.tofile(f) 
    f.close()

def Plot_specificity(Sort1, Sort2, tsf):

    comparesort = np.genfromtxt(tsf.sim_dir+Sort1.filename+'/comparesort_vs_'+Sort2.filename+'_'+
    tsf.tsf_name+'.csv', dtype='int32', delimiter=",")

    conf_arr = comparesort
    conf_arr = conf_arr.T
    max_index = conf_arr.argmax(axis=1)
    temp_array = conf_arr

    print max_index

    final_array = []
    final_index = []

    for i in range(len(conf_arr[0])): #Loop over each cell
        for j in range(len(conf_arr)): #Loop over each unit
            if max_index[j]==i:
                final_array.append(conf_arr[j])
                #print conf_arr[j]
                final_index.append(j)

    #print np.array(final_array)
    conf_arr = np.array(final_array)
    #print conf_arr
    conf_arr=conf_arr.T

    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = max(sum(i, 0),1)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    #print "NORM CONF: ", norm_conf

    ave_array=[]
    for i in range (len(norm_conf)): 
        ave_array.append(max(norm_conf[i]))

    print ave_array
    print np.mean(np.array(ave_array), axis=0)

    fig = plt.figure()
    plt.clf()
    ax = plt.subplot(1,1,1)
    #ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.YlOrRd,   #TRY BLACK TO WHITE COLOUR CHART; OR BLUE TO WHITE
                    interpolation='None')

    width = len(conf_arr[0])
    height = len(conf_arr)
    print height, width

    #for x in xrange(height):
        #for y in xrange(width):
            #ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        #horizontalalignment='center',
                        #verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = range(0,height,1) 
    
    #Y-axis should contain PTP heights
    temp_array= np.loadtxt(Sort1.directory + Sort1.name+'/'+'ptps.csv',dtype=np.float32) #Convert to seconds
    temp_array=np.array(temp_array,dtype=np.int32)
    temp_array = ['{:.2f}'.format(x) for x in temp_array]
    print temp_array
    b = [str(x) for x in alphabet]
    print b
    a = temp_array

    temp_array = []
    for i in range(len(b)):
        temp_array.append(a[i]+' - ' + b[i])
        
    plt.yticks(range(height), temp_array,fontsize=6)
    
    alphabet = final_index 
    plt.xticks(range(width), alphabet[:width],fontsize=5, fontweight='bold')

    plt.ylabel("# cells: " + str(height), fontsize=15)
    plt.xlabel("# units: " + str(width), fontsize=15)

    plt.suptitle('Specificity:  '+ Sort1.filename + '  vs  ' + Sort2.filename, fontsize = 22)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

def Plot_fpfn(sim_dir, sorted_dirs, sorted_file, anonymous):

    colors=['black','blue','green','cyan','red', 'magenta',  'pink', 'brown']

    #LOAD FILE NAMES ONLY
    file_dir='/media/cat/4TB/SFN_2015_poster/'
    if sorted_dirs == 'all':
        file_name = ['cat_sort/', 'dan_sort/', 'nicksw_sort/', 'nickst_sort/']
    else:
        file_name = sorted_dirs

    #NUMBER OF COMPARISON SORTS
    n_sorts = len(file_name)
    FP=[]
    FP_x=[]
    n_units=[]
    for i in range(n_sorts):
        FP.append([])
        FP_x.append([])
        n_units.append([])

    for i in range(len(file_name)):
        print "Loading: ", file_name[i]
        FP[i]=np.loadtxt(sim_dir+file_name[i]+'FP_histogram_' + sorted_file+ '.csv', delimiter=",")
        FP_x[i]=np.loadtxt(sim_dir+file_name[i]+'FP_histogram_x_' + sorted_file+'.csv', delimiter=",")

    ax=plt.subplot(1,1,1)
    for i in range(len(file_name)):
        plt.plot(FP_x[i],FP[i]*1E2,color=colors[i], linewidth=5)
        plt.scatter(FP_x[i],FP[i]*1E2, s=65,color=colors[i])

    #ax.set_xlim(0,500)
    #ax.set_ylim(0,100)

    ax.tick_params(axis='both', which='major', labelsize=65)
    ax.tick_params(axis='both', which='minor', labelsize=65)
    #ax.set_title('Error Rate (10uV bins)' + sorted_file, fontsize = 65)

    #Plot Legend
    black_patch = mpatches.Patch(color='black')
    blue_patch = mpatches.Patch(color='blue')       #Dan Denman         #2
    green_patch = mpatches.Patch(color='green')     #     #4
    brown_patch = mpatches.Patch(color='brown')     #                   #6
    red_patch = mpatches.Patch(color='red')         #      #3
    magenta_patch = mpatches.Patch(color='magenta') #      #5
    cyan_patch = mpatches.Patch(color='cyan')       #                   #7
    pink_patch = mpatches.Patch(color='pink')   #                   #8

    if anonymous: 
        labels = ['Op #1', 'Op #2', 'Op #3', 'Op #4', 'Op #5', 'Op #6', 'Op #7'][0:len(sorted_dirs)]
    else:
        labels = sorted_dirs

    ax.legend([black_patch, blue_patch, green_patch, cyan_patch, red_patch, magenta_patch, pink_patch ], labels, fontsize=20, loc=0, 
    title="Operator")
    bin_width = 10
    #plt.suptitle("Sorted data: " + sorted_file, fontsize = 30, fontweight='bold' )
    #plt.title("Binned False Positive rate ("+str(bin_width)+" uV bins; only bins with data)", 
    #fontsize=25)

    plt.ylim(0,100)
    plt.xlim(0,400)
    plt.ylabel("Percent Error Rate (10uV bins)", fontsize=40)
    plt.xlabel("Peak-to-peak amplitude of sorted units (uV)", fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=40)
    plt.tick_params(axis='both', which='minor', labelsize=40)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    #plt.subplots_adjust(left=0.07, right=0.93, top=0.9, bottom=0.1)
    plt.show()
    return None 
    
    #************************** FN Rates *************
    ax=plt.subplot(1,1,1)
    for i in range(len(file_name)):
        plt.plot(FP_x[i],FP[i],color=colors[i], linewidth=5)
        plt.scatter(FP_x[i],FP[i], s=65,color=colors[i])
        
    ax=plt.subplot(1,1,1)
    for i in [0,1]:
        plt.plot(x,FN[i],color=colors[i], linewidth=20)
    ax.set_xlim(0,500)
    ax.set_ylim(0,1)

    ax.tick_params(axis='both', which='major', labelsize=45)
    ax.tick_params(axis='both', which='minor', labelsize=45)
    ax.set_title('False Negative Rate', fontsize = 45)
    ax.set_xlabel('Peak-to-peak of sorted units (50uV bins)', fontsize = 45)

    plt.show()

    quit()


    ax=plt.subplot(1,1,1)
    for i in [0,3]:
        plt.plot(x,n_units[i][0:11],color=colors[i],linewidth=20)
    ax.set_xlim(0,500)
    #ax.set_ylim(0,1)

    ax.tick_params(axis='both', which='major', labelsize=45)
    ax.tick_params(axis='both', which='minor', labelsize=45)
    ax.set_title('False Positive Rate', fontsize = 45)
    ax.set_xlabel('Peak-to-peak of sorted units (50uV bins)', fontsize = 45)
    ax.set_title('No. of Units Sorted', fontsize = 100)
    ax.set_xlabel('Peak-to-peak of sorted units (50uV bins)', fontsize = 50)

    ax.set_ylabel('No. of Sorted Units', fontsize = 50)
    plt.show()






    #********************** PLOT SPECIFICITY ***************************
    
    #Sort1 = sorted data       #NOTE THIS IS INVERTED TO PLOTS BELOW as the "comparesort" file is different
    #Sort2 = ground truth data
    
    comparesort = np.genfromtxt(tsf.sim_dir+'/comparesort_'+Sort1.filename+'_vs_'+
    Sort2.filename+'.csv', dtype='int32', delimiter=",")
    
    size = np.genfromtxt(tsf.sim_dir+Sort1.filename+'_size.csv', dtype='int32', delimiter=",")
    size2 = np.genfromtxt(tsf.sim_dir+Sort2.filename+'_size.csv', dtype='int32', delimiter=",")
    
    ptp = np.genfromtxt(tsf.sim_dir+Sort1.filename+'_ptps.csv', dtype='int32', delimiter=",")
    ptp2 = np.genfromtxt(tsf.sim_dir+Sort2.filename+'_ptps.csv', dtype='int32', delimiter=",")

    completeness=[]
    purity=[]

    for i in range(len(size)):  #loop over # units
        #missed spikes from principal cell / total spikes in principal cell
        best_match = np.argmax(comparesort[i])
        completeness.append(1-float(size2[best_match] - max(comparesort[i]))/float(size2[best_match]))

        #spikes from non-principal cell / total spikes in unit
        purity.append(1-float(sum(comparesort[i])-max(comparesort[i]))/float(sum(comparesort[i]))) #Add 1 to avoid null division

    #COMPLETENESS PLOT
    plt.suptitle(tsf.tsf_name,fontsize=20)
    ax1 = plt.subplot(1,2,1)
    ax1.scatter(ptp,completeness, color='blue', s=100)
    ax1.set_title('Completeness (1 - False Negative)', fontsize = 22)
    ax1.set_ylim(0,1)
    ax1.set_xlabel(' PTP of Best Matching Cell ')
    
    ##Sort data in order of ptp
    #fit_x=ptp
    #fit_y=completeness
    #fit_y = [x for (y,x) in sorted(zip(fit_x,fit_y))]
    #fit_x = np.sort(np.array(fit_x))
    
    ##bin data in order 
    #x=[]
    #y=[]
    #bin_width=5
    #for i in range(int(len(ptp)/bin_width)):
        #temp_x=0
        #temp_y=0
        #for j in range(bin_width):
            #temp_x +=fit_x[i*bin_width+j]
            #temp_y +=fit_y[i*bin_width+j]
        #x.append(float(temp_x)/float(bin_width))
        #y.append(float(temp_y)/float(bin_width))

    #ax1.plot(x,y, linewidth=3, color='blue')
    
    #np.savetxt(tsf.sim_dir+Sort1.filename+'_completeness_x.csv', x, delimiter=",")
    #np.savetxt(tsf.sim_dir+Sort1.filename+'_completeness_y.csv', y, delimiter=",")

    #np.savetxt(tsf.sim_dir+Sort1.filename+'_scattercompleteness_x.csv', ptp, delimiter=",")
    #np.savetxt(tsf.sim_dir+Sort1.filename+'_scattercompleteness_y.csv', completeness, delimiter=",")
  
    #PURITY PLOT
    ax2 = plt.subplot(1,2,2)
    ax2.scatter(ptp,purity, color='red', s=100)
    ax2.set_ylim(0,1)
    ax2.set_title('Purity (1 - False Positive)', fontsize = 22)
    ax2.set_xlabel(' PTP of Unit ')

    ##Sort data in order of ptp
    #fit_x=ptp
    #fit_y=purity
    #fit_y = [x for (y,x) in sorted(zip(fit_x,fit_y))]
    #fit_x = np.sort(np.array(fit_x))
    
    ##bin data in order 
    #x=[]
    #y=[]
    #for i in range(int(len(ptp)/bin_width)):
        #temp_x=0
        #temp_y=0
        #for j in range(bin_width):
            #temp_x +=fit_x[i*bin_width+j]
            #temp_y +=fit_y[i*bin_width+j]
        #x.append(float(temp_x)/float(bin_width))
        #y.append(float(temp_y)/float(bin_width))

    #ax2.plot(x,y, linewidth=3, color='red')

    #np.savetxt(tsf.sim_dir+Sort1.filename+'_purity_x.csv', x, delimiter=",")
    #np.savetxt(tsf.sim_dir+Sort1.filename+'_purity_y.csv', y, delimiter=",")

    #np.savetxt(tsf.sim_dir+Sort1.filename+'_scatterpurity_x.csv', ptp, delimiter=",")
    #np.savetxt(tsf.sim_dir+Sort1.filename+'_scatterpurity_y.csv', purity, delimiter=",")
    
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    #plt.suptitle(Sort1.name + "  " + Sort2.name, fontsize = 22)

    plt.show()

    work_dir = '/media/cat/4TB/in_silico/ucsd_July5/'
    colors=['blue','red','green']
    ax1 = plt.subplot(1,2,1)
    for i in range(3):
        x = np.genfromtxt(work_dir+'ECP_'+str(i)+'_noise/ECP_'+str(i)+'_noise_completeness_x.csv', delimiter=",")
        y = np.genfromtxt(work_dir+'ECP_'+str(i)+'_noise/ECP_'+str(i)+'_noise_completeness_y.csv', delimiter=",")
        
        plt.plot(x,y,linewidth=3,color=colors[i])
        
    ax2 = plt.subplot(1,2,2)
    for i in range(3):
        x = np.genfromtxt(work_dir+'ECP_'+str(i)+'_noise/ECP_'+str(i)+'_noise_purity_x.csv', delimiter=",")
        y = np.genfromtxt(work_dir+'ECP_'+str(i)+'_noise/ECP_'+str(i)+'_noise_purity_y.csv',delimiter=",")

        plt.plot(x,y,linewidth=3,color=colors[i])
        
    plt.show()


def Compute_FP_histograms(Sort1, Sort2, tsf):

    comparesort = np.genfromtxt(Sort2.directory+'/comparesort_'+Sort1.filename+'_vs_'+
    Sort2.filename+'.csv', dtype='int32', delimiter=",")
    
    size = np.genfromtxt(Sort1.directory+Sort1.filename+'_size.csv', dtype='int32', delimiter=",")
    size2 = np.genfromtxt(Sort2.directory+Sort2.filename+'_size.csv', dtype='int32', delimiter=",")
    ptps = np.genfromtxt(Sort2.directory+Sort2.filename+'_ptps.csv', dtype='int32', delimiter=",")

    conf_arr = comparesort
    max_index = conf_arr.argmax(axis=1) #Find best match for each cell by searching for max spikes unit
    conf_arr = conf_arr.T               #Transpose the conf_array to rearrange by columns in order of unit matches

    #Rearrange conf_matrix columns bringing matching units match order of matching cells
    final_array = np.zeros((len(size2),len(size)), dtype=np.int32)
    final_index = []
    for i in range(len(size2)): #Loop over each cell #NB: Can't have more sorted cells than ground truth cells!!
        final_array[i]=conf_arr[max_index[i]]
        final_index.append([max_index[i]])

    #print final_array.shape
    conf_arr = final_array.T

    #NOTE: SPECIFICITY measures how "specific" a unit is to a cell: i.e. % of unit spikes from matching cell 
    # divided by the total spikes in that unit; This is essentiality PURITY
    norm_conf = []
    counter=0
    for i in conf_arr:
        a = 0
        tmp_arr = []
        #aa = np.array(i).argmax()
        #a = max(sum(i, 0),1)  ; method = '(norm: detected cell spikes)'     #Normalized to total detected spikes from cell
        a = size2[max_index[counter]]  ; method = '(norm: total spikes in unit)'  #Normalized to total spikes in unit
        #a = size[counter]  ; method = '(norm: total cell spikes)'              #Sensitivity plot: Normalized to total spikes in cell (regardless of detection)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
        #THIS conditional searches for poor matches for very large units
        #if max(tmp_arr)<0.50 and (ptps[max_index[counter]]>300):
            #print tmp_arr
            #print ptps[max_index[counter]]
            #print "unit: ", max_index[counter]
            #print "cell: ", counter
            ##quit()
        counter+=1

    FP_matrix = 1-np.array(norm_conf) #[rows, columns]
    norm_conf = FP_matrix 

    n_units=len(norm_conf[0])

    #print "PTPS_array:", ptps_array

    #Plot binned FP rates; Compute average FP rate in uV blocks:
    ax=plt.subplot(1,1,1)
    FP_histogram=[]
    FP_histogram_x=[] #Keep track of which bins have values
    bin_width = 10 #size of block in uV
    first_bin = 10
    counter=0
    #n_cells=np.zeros(100, dtype=np.int32) #no. of cells in each bin
    for i in np.arange(first_bin, 650, bin_width):
        temp=0.
        counter=0
        for j in range(len(FP_matrix[0])):
            if (ptps[j]>i) and (ptps[j]<i+bin_width):
                counter+=1
                temp+=FP_matrix[j][j] #collect FP rates
        if counter>0:
            FP_histogram.append(float(temp)/float(counter))  #Divide cumulative percentage by number of cells in bin
            FP_histogram_x.append(i)
    
        np.savetxt(Sort2.directory+'/FP_histogram_'+Sort2.filename+'.csv', FP_histogram, delimiter=",")
    np.savetxt(Sort2.directory+'/FP_histogram_x_'+Sort2.filename+'.csv', FP_histogram_x, delimiter=",")
        

def Plot_confusion(Sort1, Sort2, tsf):

    #print "PLOTTING CONFUSION MATRIX"

    #********************** PLOT SPECIFICITY ***************************
    
    comparesort = np.genfromtxt(Sort2.directory+'/comparesort_'+Sort1.filename+'_vs_'+
    Sort2.filename+'.csv', dtype='int32', delimiter=",")
    
    size = np.genfromtxt(Sort1.directory+Sort1.filename+'_size.csv', dtype='int32', delimiter=",")
    size2 = np.genfromtxt(Sort2.directory+Sort2.filename+'_size.csv', dtype='int32', delimiter=",")
    ptps = np.genfromtxt(Sort2.directory+Sort2.filename+'_ptps.csv', dtype='int32', delimiter=",")

    conf_arr = comparesort
    max_index = conf_arr.argmax(axis=1) #Find best match for each cell by searching for max spikes unit
    conf_arr = conf_arr.T               #Transpose the conf_array to rearrange by columns in order of unit matches

    #Rearrange conf_matrix columns bringing matching units match order of matching cells
    final_array = np.zeros((len(size2),len(size)), dtype=np.int32)
    final_index = []
    for i in range(len(size2)): #Loop over each cell #NB: Can't have more sorted cells than ground truth cells!!
        print i
        final_array[i]=conf_arr[max_index[i]]
        final_index.append([max_index[i]])

    #print final_array.shape
    conf_arr = final_array.T

    #NOTE: SPECIFICITY measures how "specific" a unit is to a cell: i.e. % of unit spikes from matching cell 
    # divided by the total spikes in that unit; This is essentiality PURITY
    norm_conf = []
    counter=0
    for i in conf_arr:
        a = 0
        tmp_arr = []
        #aa = np.array(i).argmax()
        #a = max(sum(i, 0),1)  ; method = '(norm: detected cell spikes)'     #Normalized to total detected spikes from cell
        a = size2[max_index[counter]]  ; method = '(norm: total spikes in unit)'  #Normalized to total spikes in unit
        #a = size[counter]  ; method = '(norm: total cell spikes)'              #Sensitivity plot: Normalized to total spikes in cell (regardless of detection)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
        #THIS conditional searches for poor matches for very large units
        #if max(tmp_arr)<0.50 and (ptps[max_index[counter]]>300):
            #print tmp_arr
            #print ptps[max_index[counter]]
            #print "unit: ", max_index[counter]
            #print "cell: ", counter
            ##quit()
        counter+=1

    FP_matrix = 1-np.array(norm_conf) #[rows, columns]
    norm_conf = FP_matrix 

    n_units=len(norm_conf[0])

    #print "PTPS_array:", ptps_array

    #Plot binned FP rates; Compute average FP rate in uV blocks:
    ax=plt.subplot(1,1,1)
    FP_histogram=[]
    FP_histogram_x=[] #Keep track of which bins have values
    bin_width = 10 #size of block in uV
    first_bin = 10
    counter=0
    #n_cells=np.zeros(100, dtype=np.int32) #no. of cells in each bin
    for i in np.arange(first_bin, 650, bin_width):
        temp=0.
        counter=0
        for j in range(len(FP_matrix[0])):
            if (ptps[j]>i) and (ptps[j]<i+bin_width):
                counter+=1
                temp+=FP_matrix[j][j] #collect FP rates
        if counter>0:
            FP_histogram.append(float(temp)/float(counter))  #Divide cumulative percentage by number of cells in bin
            FP_histogram_x.append(i)
    
    
    #np.savetxt(Sort2.directory+'/FP_histogram_'+Sort2.filename+'.csv', FP_histogram, delimiter=",")
    #np.savetxt(Sort2.directory+'/FP_histogram_x_'+Sort2.filename+'.csv', FP_histogram_x, delimiter=",")
    
    
    x = np.arange(first_bin,650,bin_width)

    #plt.bar(FP_histogram_x,FP_histogram,width=bin_width-3, color='red')

    plt.scatter(FP_histogram_x,FP_histogram, s=100,color='black')
    plt.plot(FP_histogram_x, FP_histogram, linewidth=3, color='black')

    plt.ylim(0,1)
    plt.xlim(0,500)
    plt.ylabel("False Positive Rate", fontsize=30)
    plt.xlabel("Peak-to-peak amplitude of sorted units (uV)", fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.tick_params(axis='both', which='minor', labelsize=3255)

    #Plot pie-charts

    black_patch = mpatches.Patch(color='black')
    blue_patch = mpatches.Patch(color='blue')       #Dan Denman         #2
    red_patch = mpatches.Patch(color='red')         #Nick Swindale      #3
    green_patch = mpatches.Patch(color='green')     #Nick Steimtz       #4
    magenta_patch = mpatches.Patch(color='magenta') #Martin Spacek      #5
    brown_patch = mpatches.Patch(color='brown')     #                   #6
    cyan_patch = mpatches.Patch(color='cyan')       #                   #7
    pink_patch = mpatches.Patch(color='pink')   #                   #8

    labels = ['Op #1', 'Op #2', 'Op #3', 'Op #4', 'Op #5', 'Op #6']

    ax.legend([black_patch, blue_patch, red_patch, green_patch, magenta_patch, pink_patch], labels, fontsize=12, loc=0, 
    title="Operator")
    


    plt.suptitle("Sorted data: " + Sort2.name, fontsize = 30, fontweight='bold' )
    plt.title("Binned False Positive rate ("+str(bin_width)+" uV bins; only bins with data)", 
    fontsize=25)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()
    return None

    #********************************** FP CONFUSION MATRIX ******************************
    #Plotting section starts here

    fig = plt.figure()
    plt.clf()
    #ax1 = plt.subplot(1,2,1)
    ax1 = plt.subplot(1,2,1)
    #ax.set_aspect(1)
    res = ax1.imshow(np.array(norm_conf), cmap=plt.cm.Reds,   #TRY BLACK TO WHITE COLOUR CHART; OR BLUE TO WHITE
                    interpolation='None')

    width = len(conf_arr[0])
    height = len(conf_arr)
    print height, width

    cb = fig.colorbar(res)
    alphabet = range(0,height,1) 
    
    #Y-axis LABELS: 
    temp_array=np.loadtxt(Sort1.directory + Sort1.name+'_ptps.csv',dtype=np.float32) #Convert to seconds

    ptps_array = np.array(temp_array, dtype=np.float32)
    
    temp_array=np.array(temp_array,dtype=np.int32)

    temp_array=['{:d}'.format(x) for x in temp_array]
    #print temp_array
    b = [str(x) for x in alphabet]
    #print b
    a = temp_array

    temp_array = []
    for i in range(len(b)):
        temp_array.append(a[i]+' - ' + b[i])

    #ax1.set_yticklabels([])
    plt.yticks(range(height), temp_array,fontsize=4)

    alphabet = final_index 
    #plt.xticks(range(width), alphabet[:width],fontsize=5, fontweight='bold')

    #plt.ylabel("Cells closest to proto probe (<50 um) matching the sorted units", fontsize=22)
    plt.ylabel(str(height) + ' cells with amplitude > 80uV \n(ordered by amplitude)', fontsize=22)

    plt.xlabel(str(width) + ' sorted units', fontsize=22)

    ax1.set_xlim(0,width)
    ax1.set_ylim(height,0)

    ax1.set_xticklabels([])
    
    ax1.set_title('False Positive Rate* \n(*incl. noise and undersplit spikes)', fontsize = 22)
    
    plt.show()
    #quit()

    #************************************ PLOTING SENSITIVITY *********************
    
    comparesort = np.genfromtxt(Sort2.directory+'/comparesort_'+Sort1.filename+'_vs_'+
    Sort2.filename+'.csv', dtype='int32', delimiter=",")

    #size = np.genfromtxt(tsf.sim_dir+Sort1.filename+'_size.csv', dtype='int32', delimiter=",")

    #NEW way
    conf_arr = comparesort
    max_index = conf_arr.argmax(axis=1) #Find best match for each cell by searching for max spikes unit
    conf_arr = conf_arr.T               #Transpose the conf_array to rearrange by columns in order of unit matches

    print max_index

    #Rearrange conf_matrix columns bringing matching units match order of matching cells
    final_array = np.zeros((len(size2),len(size)), dtype=np.int32)
    final_index = []
    for i in range(len(size2)): #Loop over each cell
        final_array[i]=conf_arr[max_index[i]]
        final_index.append([max_index[i]])

    print final_array.shape
    conf_arr = final_array.T 

    norm_conf = []
    counter=0
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = size[counter] #max(sum(i, 0),1) #Cat: change the denominator for sensitivity
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
        counter+=1

    FP_matrix = 1-np.array(norm_conf) #[rows, columns]
    norm_conf = FP_matrix #Invert array to plot false positive rates:

    #Compute average FP rate in 50uV blocks:
    FP_histogram=[]
    bin_width = 25
    counter=0
    n_cells=np.zeros(15,dtype=np.int32)
    for i in np.arange(25,650,bin_width):
        temp=0.
        counter=0
        for j in range(n_units):
            if (ptps_array[j]>i) and (ptps_array[j]<i+bin_width):
                counter+=1
                temp+=FP_matrix[j,j]
        if counter>0:
            FP_histogram.append(float(temp)/float(counter))
        else:
            FP_histogram.append(0.)
        n_cells[i/50]=counter

    np.savetxt(Sort2.directory+'/FN_histogram_'+Sort2.filename+'.csv', FP_histogram, delimiter=",")
    np.savetxt(Sort2.directory+'/FN_histogram_n_units_'+Sort2.filename+'.csv', n_cells, delimiter=",")
    
    x = np.arange(25,650,bin_width)
    plt.bar(x,FP_histogram,width=bin_width-3, color='red')
    plt.ylim(0,1)
    plt.xlim(0,500)
    plt.ylabel("False Negative Rate", fontsize=55)
    plt.xlabel("Peak-to-peak amplitude of sorted units (uV)", fontsize=55)
    plt.tick_params(axis='both', which='major', labelsize=40)
    plt.tick_params(axis='both', which='minor', labelsize=40)
    
    plt.text(20,.17,'50um',fontsize=20, fontweight='bold')
    plt.text(12.5,.26,'50um',fontsize=20, rotation='vertical',fontweight='bold')

    
    plt.title(Sort1.filename, fontsize=35)
    
    plt.show()
    
    #quit()


    #********************************** FN CONFUSION MATRIX ******************************

    #plt.clf()
    #fig = plt.figure()
    ax2 = plt.subplot(1,2,2)
    #ax.set_aspect(1)
    res = ax2.imshow(np.array(norm_conf), cmap=plt.cm.Blues,   #TRY BLACK TO WHITE COLOUR CHART; OR BLUE TO WHITE
                    interpolation='nearest')

    width = len(conf_arr[0])
    height = len(conf_arr)
    print height, width

    #for x in xrange(height):
        #for y in xrange(width):
            #ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        #horizontalalignment='center',
                        #verticalalignment='center')

    #cb = fig.colorbar(res)
    alphabet = range(0,height,1) 

    #Y-axis should contain PTP heights
    temp_array= np.loadtxt(Sort1.directory + Sort1.name+'_ptps.csv',dtype=np.float32) #Convert to seconds
    temp_array=np.array(temp_array,dtype=np.int32)
    temp_array = ['{:.2f}'.format(x) for x in temp_array]
    #print temp_array
    b = [str(x) for x in alphabet]
    #print b
    a = temp_array

    temp_array = []
    for i in range(len(b)):
        temp_array.append(a[i]+' - ' + b[i])

    #plt.yticks(range(height), temp_array,fontsize=5)
    alphabet = final_index 
    #plt.xticks(range(width), alphabet[:width],fontsize=5)
    
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    plt.ylabel("# cells < 50um from probe: " + str(height) + ' \n (ordered by amplitude)', fontsize=22)
    plt.xlabel("# sorted units: " + str(width) + ' \n(order of matching cell)', fontsize=22)

    ax2.set_title('False negative Rate', fontsize = 22)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    
    plt.suptitle(Sort1.name + "  " + Sort2.name, fontsize = 22)

    plt.show()

def Plot_table(Sort1, Sort2, tsf):

    comparesort = np.genfromtxt(tsf.sim_dir+Sort1.filename+'/comparesort_vs_'+Sort2.filename+'_'+
    tsf.tsf_name+'.csv', dtype='int32', delimiter=",")

    matplotlib.rc('text', usetex=True)
    plt.figure()
    ax=plt.gca()
    y=[1,2,3,4,5,4,3,2,1,1,1,1,1,1,1,1]
    #plt.plot([10,10,14,14,10],[2,4,4,2,2],'r')
    col_labels=['col1','col2','col3']
    row_labels=['row1','row2','row3']
    table_vals=[11,12,13,21,22,23,31,32,33]
    table = r'''\begin{tabular}{ c | c | c | c } & col1 & col2 & col3 \\\hline row1 & 11 & 12 & 13 \\\hline row2 & 21 & 22 & 23 \\\hline  row3 & 31 & 32 & 33 \end{tabular}'''
    plt.text(9,3.4,table,size=12)
    plt.plot(y)
    plt.show()


def find_nearest(array,value,dist):
    #print "Looking for: ", value
    #print "Diff to closest value: ", array[np.abs(array-value).argmin()]-value
    if abs(array[np.abs(array-value).argmin()]-value) < dist:
        idx = (np.abs(array-value)).argmin()
        return idx
    else:
        return None

def find_nearest_forward(array,value,dist):
    #Modified version of above, only searches for positive (i.e. forward) values

    if abs(array[np.abs(array-value).argmin()]-value) < dist:
        idx = (np.abs(array-value)).argmin()
        return idx
    else:
        return None

def fit_log(x,a,b,c,d):
    return a + b*np.log(100-1/x)

def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)

def exponential(x,a,b,c,d):
    return a*np.exp(b*x+c)+d

def sigmoid(p,x):
    x0,y0,c,k=p
    y = c / (1 + np.exp(-k*(x-x0))) + y0
    return y

def sigmoid2(x, x0, k):
    y = 100 / (1 + np.exp(-k*(x-0.5)))
    return y 

#def sigmoid3(x, a, b,c,d):
    #y = 1-np.exp((-(x-a)/b)**c) #  + b/x
    #return y

def asymptoticx(x, a, b,c,d):
    y =a+b*x/(x+d) #+ /x
    return y

def asymptoticlogx(x, a, b,c,d,e,f,g):
    y =a+ b*np.log(x+d) #/(1+log(x+f)) #+ /x
    return y


def sigmoid_function(xdata, x0, k):
    y = np.exp(-k*(xdata-x0)) / (1 + np.exp(-k*(xdata-x0)))
    return y

def residuals(p,x,y):
    return y - sigmoid(p,x)

def resize(arr,lower=0.0,upper=1.0):
    arr=arr.copy()
    if lower>upper: lower,upper=upper,lower
    arr -= arr.min()
    arr *= (upper-lower)/arr.max()
    arr += lower
    return arr

def draw_pie(ax,ratios=[0.4,0.3,0.3], X=0, Y=0, size = 1000, colors=colors):
    N = len(ratios)
 
    xy = []
 
    start = 0.
    for ratio in ratios:
        x = [0] + np.cos(np.linspace(2*math.pi*start,2*math.pi*(start+ratio), 30)).tolist()
        y = [0] + np.sin(np.linspace(2*math.pi*start,2*math.pi*(start+ratio), 30)).tolist()
        xy1 = zip(x,y)
        xy.append(xy1)
        start += ratio
 
    for i, xyi in enumerate(xy):
        ax.scatter([X],[Y] , marker=(xyi,0), s=size, facecolor=colors[i])

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

def gauss(x, *p):
    A, mu, S = p
    return A*np.exp(-(x-mu)**2/(2.*S**2))

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def wavelet(data, wname="db4", maxlevel=6):
    """Perform wavelet multi-level decomposition and reconstruction (WMLDR) on data.
    See Wiltschko2008. Default to Daubechies(4) wavelet"""
    import pywt

    data = np.atleast_2d(data)
    # filter data in place:
    for i in range(len(data)):
        print "Wavelet filtering channel: ", i
        # decompose the signal:
        c = pywt.wavedec(data[i], wname, level=maxlevel)
        # destroy the appropriate approximation coefficients:
        c[0] = None
        # reconstruct the signal:
        data[i] = pywt.waverec(c, wname)
    
    return data

def Concatenate_LFP_sort(sim_dir, recordings, track_name):
    '''Concatenate LFP track from Martin's lfp data for LFP sorting
    '''

    import glob

    lfp = Object_temp()
    lfp.data = []       #LFP data
    lfp.t_start = []    #Start of lfp data relative high-pass record
    lfp.t_end = []      #end of lfp data
    #lfp.rec = []        #recording index (string) 
    #lfp.time_index = [] #running time index of data - need in order to index into LFP event data file
    #lfp.rec_name = []   #holds names or recordings
    lfp.sampfreq = 1000
    
    counter=0
    print "Making compressed LFP file..."
    for recording in recordings:
        print "Loading lfp zip: ", recording
        temp_name = glob.glob(sim_dir+recording+"*")[0]
        file_name = temp_name.replace(sim_dir, '')
        data_in  = np.load(sim_dir+file_name+'/'+file_name+'.lfp.zip')
        lfp.t_start.append(data_in['t0']*1E-6)      #Convert to seconds
        lfp.t_end.append(data_in['t1']*1E-6)

        #Offset start of data to t = t0
        temp_data = data_in['data']
        start_offset = np.zeros((len(data_in['data']), lfp.t_start[counter]*1E3), dtype=np.int16)
        data_out = np.concatenate((start_offset, temp_data), axis=1)
        
        #Remove 60Hz frequency
        for k in range(len(data_out)):
            data_out[k] = np.array(filter.notch(data_out[k])[0], dtype=np.int16) # remove 60 Hz mains noise
        #data_out = temp_data

        #Zero out sync index periods below a treshold; 
        si_limit=0.7            
        si, t, sync_periods = synchrony_index(data_out[9], lfp, counter, si_limit)
        print "sync periods: ", sync_periods
        
        temp_data = data_out.copy() * 0     #Make a copy of original data - zero out all data;
        for j in range(len(sync_periods)):      #Loop over sync periods
            for k in range(len(data_out)):      #Loop over all channels
                #print len(data_out[k]), sync_periods[j][0]*1E3, sync_periods[j][1]*1E3
                temp_data[k][sync_periods[j][0]*1E3:sync_periods[j][1]*1E3]=data_out[k][sync_periods[j][0]*1E3:sync_periods[j][1]*1E3]
        data_out = temp_data
        
        if counter==0: lfp.data = data_out
        else: lfp.data = np.concatenate((lfp.data, data_out), axis=1)

        counter+=1
        
    print lfp.data.shape

    #quit()


    #Make concatenated compressed lfp file into .tsf 
    subsample = 10
    file_name = sim_dir + track_name + '_notch_'+str(subsample)+'subsample.tsf'
    header = 'Test spike file '
    iformat = 1002
    vscale_HP = data_in['uVperAD']
    SampleFrequency = 20000
    n_electrodes = len(lfp.data)
    n_vd_samples = len(lfp.data[0][::subsample])       #Compression factor
    print "Total recording length: ", n_vd_samples 

    f = open(file_name, 'wb')
    f.write(header)
    f.write(struct.pack('i', iformat))
    f.write(struct.pack('i', SampleFrequency))
    f.write(struct.pack('i', n_electrodes))
    f.write(struct.pack('i', n_vd_samples))
    f.write(struct.pack('f', vscale_HP))

    for i in range (n_electrodes):
        print "Writing electrode: ", i
        f.write(struct.pack('h', 0))
        f.write(struct.pack('h', i*100))
        f.write(struct.pack('i', i+1))

    print "Writing data"

    for i in range (n_electrodes):
        print "Writing channel: ", i
        #data = np.array(lfp_traces[i],dtype=np.int16) # remove 60 Hz mains noise, as for SI calc
        lfp.data[i][::subsample].tofile(f)

    f.write(struct.pack('i', 0)) #Write zero spikes
    f.close()
    
    print "Saved: ", file_name
    
    quit()

    #Make regular non-compressed .tsf file
    file_name = sim_dir + "track"+track_name[-2:] + '_uncompressed_'+str(lowcut)+"hz_"+str(highcut)+"hz.tsf"
    header = 'Test spike file '
    iformat = 1002
    vscale_HP = 1.
    n_vd_samples = len(lfp.lfp_traces[0])
    print "Total uncompressed rec length: ", n_vd_samples 

    f = open(file_name, 'wb')
    f.write(header)
    f.write(struct.pack('i', iformat))
    f.write(struct.pack('i', lfp.SampleFrequency))
    f.write(struct.pack('i', lfp.n_electrodes))
    f.write(struct.pack('i', n_vd_samples))
    f.write(struct.pack('f', vscale_HP))

    for i in range (lfp.n_electrodes):
        f.write(struct.pack('h', 0))
        f.write(struct.pack('h', i*100))
        f.write(struct.pack('i', i+1))

    print "Writing data"

    for i in range (lfp.n_electrodes):
        print "Writing channel: ", i
        data = np.array(lfp.lfp_traces[i],dtype=np.int16) # remove 60 Hz mains noise, as for SI calc
        data.tofile(f)

    f.write(struct.pack('i', 0)) #Write zero spikes
    f.close()
    
    print "Saved: ", file_name
        


def Concatenate_LFPtraces_old(sim_dir, track_name, recordings):

    import re

    files = os.listdir(sim_dir)
    def natural_key(string_):
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
    files = sorted(files, key=natural_key)

    lfp = type('lfp', (object,), {})()
    compressed_lfp = type('lfp', (object,), {})()
    
    lfp.lfp_traces = []
    lfp.lfp_length = []
    compressed_lfp.lfp_traces = []
    for i in range(10):
        lfp.lfp_traces.append([])
        compressed_lfp.lfp_traces.append([])
        
    lfp.n_electrodes = 10
    compressed_lfp.n_electrodes = 10
    lfp.header = ''
    compressed_lfp.header = ''

    sub_sample_rate = 5     #Subsample LFP traces to make it easier for SS to disaply an dprocess

    compress_rate = 100     #Temporal compression for spike sorting

    lowcut = 1
    highcut = 240

    non_linear = False

    recording_lengths = []

    lfp.SampleFrequency = 1000
    compressed_lfp.SampleFrequency = 1000 * compress_rate/sub_sample_rate #1KHZ LFP SAMPLE RATE for Nick's data * compression factor;

    for file_name in files:
        for k in range(len(recordings)):
            #print str(recordings[k]) + '-' + track_name
            
            if ((str(recordings[k]) + '-' + track_name) in file_name) or ((str(recordings[k])=='all') and (track_name in file_name)):
                
                print "loading: ", file_name

                temp_data = np.load(sim_dir+file_name+'/'+file_name+'.lfp.zip')

                compressed_lfp.vscale = temp_data['uVperAD']
                for i in range(10):
                    data = temp_data['data'][i]*compressed_lfp.vscale
                    
                    #Apply notch filter
                    if True:
                        data = np.array(filter.notch(data)[0],dtype=np.int16) # remove 60 Hz mains noise, as for SI calc

                    ##Apply butterworth filter; keep frequencies between:  lowcut < XX < highcut
                    if True:
                        fs = 1000 #Original sampling frequency for Nick's data is 1Khz
                        data = np.array(butter_bandpass_filter(data, lowcut, highcut, fs, order = 2), dtype=np.int16)

                    #Save notch filtered/etc. data to non-compressed lfp traces; first add blank space, then data;
                    blank_length = temp_data['t0']/1E3  #Calculate length of blank recording in ms, i.e. 1Khz sample rate;
                    blank_data = np.zeros(blank_length,dtype=np.int16)
                    lfp.lfp_traces[i].extend(blank_data)
                    lfp.lfp_traces[i].extend(data)
                    
                    #Subsample data 
                    data = data[0::sub_sample_rate]

                    #if non_linear:
                        ###Apply exp nonlinearity
                        #exp_array = np.zeros(len(data),dtype=np.float32)+1.71
                        #data = np.float32(data)/20.
                        #data = np.power(exp_array, data)
                        
                        ##data = np.exp(np.float32(data)/20.)

                        #data = data/(25./lowcut)
                        ##data = np.clip(data, -1000, 1000)
                        
                        ###Apply sigmoid transform to data; first normalize data to 0..1
                        ##from scipy.special import expit
                        ##data = np.float32(data) + abs(min(data))    #Shift to 0..
                        ##data = data / max(data)                     #Normalize 0..1
                        ##data = expit(data) *1000
                        ###data = data - min(data)


                    #Insert blank recording time at front of data structure
                    blank_length = temp_data['t0']/1E3/sub_sample_rate  #Calculate length of blank recording in ms, i.e. 1Khz sample rate;
                    blank_data = np.zeros(blank_length,dtype=np.int16)
                    compressed_lfp.lfp_traces[i].extend(blank_data)
                    
                    compressed_lfp.lfp_traces[i].extend(data)
                                      
                recording_lengths.append(len(temp_data['data'][0])/sub_sample_rate+blank_length) #Keep track of 
                
    compressed_lfp.lfp_traces = np.array(compressed_lfp.lfp_traces, dtype=np.int16)

    #Make concatenated compressed lfp file into .tsf 
    file_name = sim_dir + "track"+track_name[-2:] + '_compressed_'+str(lowcut)+"hz_"+str(highcut)+"hz.tsf"
    header = 'Test spike file '
    iformat = 1002
    vscale_HP = 1.
    n_vd_samples = sum(recording_lengths)
    print "Total recording length: ", n_vd_samples 

    f = open(file_name, 'wb')
    f.write(header)
    f.write(struct.pack('i', iformat))
    f.write(struct.pack('i', compressed_lfp.SampleFrequency))
    f.write(struct.pack('i', compressed_lfp.n_electrodes))
    f.write(struct.pack('i', n_vd_samples))
    f.write(struct.pack('f', vscale_HP))

    for i in range (compressed_lfp.n_electrodes):
        print "Writing electrode: ", i
        f.write(struct.pack('h', 0))
        f.write(struct.pack('h', i*100))
        f.write(struct.pack('i', i+1))

    print "Writing data"

    for i in range (compressed_lfp.n_electrodes):
        print "Writing channel: ", i
        data = np.array(compressed_lfp.lfp_traces[i],dtype=np.int16) # remove 60 Hz mains noise, as for SI calc
        
        data.tofile(f)

    f.write(struct.pack('i', 0)) #Write zero spikes
    f.close()
    
    print "Saved: ", file_name
    
    #Make regular non-compressed .tsf file
    file_name = sim_dir + "track"+track_name[-2:] + '_uncompressed_'+str(lowcut)+"hz_"+str(highcut)+"hz.tsf"
    header = 'Test spike file '
    iformat = 1002
    vscale_HP = 1.
    n_vd_samples = len(lfp.lfp_traces[0])
    print "Total uncompressed rec length: ", n_vd_samples 

    f = open(file_name, 'wb')
    f.write(header)
    f.write(struct.pack('i', iformat))
    f.write(struct.pack('i', lfp.SampleFrequency))
    f.write(struct.pack('i', lfp.n_electrodes))
    f.write(struct.pack('i', n_vd_samples))
    f.write(struct.pack('f', vscale_HP))

    for i in range (lfp.n_electrodes):
        f.write(struct.pack('h', 0))
        f.write(struct.pack('h', i*100))
        f.write(struct.pack('i', i+1))

    print "Writing data"

    for i in range (lfp.n_electrodes):
        print "Writing channel: ", i
        data = np.array(lfp.lfp_traces[i],dtype=np.int16) # remove 60 Hz mains noise, as for SI calc
        data.tofile(f)

    f.write(struct.pack('i', 0)) #Write zero spikes
    f.close()
    
    print "Saved: ", file_name
        

def Plot_trackwide_specgrams(sim_dir, track_name):

    files = os.listdir(sim_dir)
    def natural_key(string_):
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
    files = sorted(files, key=natural_key)

    lfp = type('lfp', (object,), {})()
    lfp.lfp_traces = []
    for i in range(10):
        lfp.lfp_traces.append([])
    
    lfp.header = ''
    
    recording_lengths = []
    x_labels = []
    rec_length = 0
    counter=0
    offset = 25. #Space between recordings
    
    lfp.SampleFrequency = 1000 #1KHZ LFP SAMPLE RATE for Nick's data
    lfp.tres = 0
    spec_ch = 9 #Choose spec channel

    images=[]
    extent = 0
    temp_float=0.

    for file_name in files:
        if track_name in file_name:
            print "loading: ", file_name

            temp_data = np.load(sim_dir+file_name+'/'+file_name+'.lfp.zip')
            print temp_data['t0']
            print temp_data['t1']

            lfp.vscale = temp_data['uVperAD']
            lfp.lfp_traces= temp_data['data']*lfp.vscale
            
            Plot_specgram(lfp, show_plot=False, spec_ch=spec_ch)
            images.append(lfp.specgram[::-1])
            
            recording_lengths.append(float(len(temp_data['data'][0]))/1000./3600.) #Convert recording length to hrs
                        
            x_labels.append(file_name)

            counter+=1
            #if counter >1: break

    print recording_lengths

    ax = plt.subplot(111)
    im = imshow(np.hstack(images), extent=[0, sum(recording_lengths),0,110],  aspect='normal') #,interpolation='none') #extent=lfp.extent, cmap=lfp.cm)
    #ax.autoscale(enable=True, tight=True)
    ax.axis('tight')

    plt.title(sim_dir + " " + track_name)
    ax.set_xlabel("Time (hrs)", labelpad=5)
    ax.set_ylabel("Frequency (Hz)")
    
    temp_array=np.zeros(len(recording_lengths),dtype=np.float32)
    temp_array[0]= recording_lengths[0]
    for i in range(1, len(recording_lengths),1):
        temp_array[i]=temp_array[i-1]+recording_lengths[i]
        
    plt.vlines(temp_array,0,110,linewidth=5, color='k',linestyles='solid')
    plt.xticks(np.arange(0, max(temp_array), .25))

    print "BLACK VLINES: ", temp_array
    
    #Print additional x axis for labels
    #recording_lengths=np.array(recording_lengths)*2.
    ax2 = ax.twiny()
    ax2.xaxis.set_ticks_position('bottom')
    ax2.tick_params(axis='x', pad=10)
    
    temp_array2=np.zeros(len(recording_lengths),dtype=np.float32)
    
    temp_array2[0]= float(recording_lengths[0])/2.
    
    for i in range(1, len(recording_lengths),1):
        temp_array2[i]= temp_array2[i-1]+recording_lengths[i-1]/2.+recording_lengths[i]/2.
    ax2.set_xlim(0, max(temp_array))
    
    print "LABELS: ", temp_array2
    plt.xticks(temp_array2, x_labels, fontweight='bold', fontsize=9, rotation = 70)


    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.20)
    fig = matplotlib.pyplot.gcf()
    print max(temp_array)
    print max(temp_array)*25
    fig.set_size_inches(max(temp_array)*20, 15)
    #print max(temp_array)*35
    fig.savefig(sim_dir+track_name, dpi=100)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

def Plot_PSTH(Sort1, sim_dir):

    ##Load stimulus times - SEV DATA
    file_name = sim_dir+Sort1.sorted_file + '/'+Sort1.sorted_file[:-7]+'.SweepTime.mat'
    mat = scipy.io.loadmat(file_name)
    stim_times = np.array(mat[mat.keys()[1]][0][0][4][0][1])/float(Sort1.samplerate)

    cumulative_triggered_trains=[]

    #BUILD HISTOGRAMS for each stimulus;
    bin_width = 0.01
    window = 0.520
    x = np.arange(0,window,bin_width)
        
    #Find all spikes in 1.0 second window after stimulus
    ave_spikes_150=[]   #Track no. of spikes in first 150ms
    ave_mean=[]         #Track average firing rate means
    for i in range(Sort1.n_units):
        cumulative_triggered_trains.append([])
        temp_list = []

        temp_array = np.array(Sort1.units[i],dtype=float32)/float(Sort1.samplerate) #WORK IN SECONDS TIMESCALE

        multi_stim_histogram=[]
        n_triggered_spikes = 0
        n_spikes_150 = 0
        #Loop over all stimulus_periods
        for j in range(len(stim_times)):
            #Find all unit spikes that fall w/in +/- window of pop spike
            temp2 = np.where(np.logical_and(temp_array>=stim_times[j][0], temp_array<=stim_times[j][0]+window))[0]
            temp3 = temp_array[temp2]-stim_times[j][0]      #Offset time of spike to t=t_0 

            cumulative_triggered_trains[i].extend(temp3) 
            
            #Compute histogram for particular stimulus response
            temp_hist = np.histogram(temp3, bins = x)[0]
            multi_stim_histogram.append(temp_hist)

            n_triggered_spikes+=len(temp3)
            n_spikes_150+= len(np.where(temp3<0.150)[0])

        #Plot individual cell histograms w. SD
        #array_std=np.std(multi_stim_histogram, axis=0)*1/bin_width #Convert values to instantaneous firing rate, i.e. per second
        array_mean=np.mean(multi_stim_histogram, axis=0)*1/bin_width
        ave_mean.append(array_mean) #Save these mean arrays for later averaging
        
        array_std=np.std(multi_stim_histogram, axis=0)*1/bin_width #Convert values to instantaneous firing rate, i.e. per second
        
        if False:
            plt.plot(x[:-1], array_mean, color='black', linewidth=3)
            plt.fill_between(x[:-1], array_mean-array_std, array_mean+array_std, facecolor='green', alpha=0.5)
            plt.ylabel("Instantenous firing rate (Hz)")
            plt.xlabel("Time from stimulus (Sec)")
            plt.title("Recording: " + sim_dir +
            "\nNo. stimuli: " + str(len(stim_times))+ ",   Length of Rec: " + str(int(stim_times[-1][1])) + " secs.   " + 
            "  Unit: "+str(i)+ "/"+str(len(Sort1.units))+ "    Total # Spikes: " + str(len(Sort1.units[i]))+ 
            "\nTotal No. spikes w/n 150 ms:  "+str(n_spikes_150) + " (" + 
            str(int(float(n_spikes_150)/float(len(Sort1.units[i]))*100)) + '%)' + "  Ave. No. spikes in first 150ms:  " +
            str(float(n_spikes_150)/float(len(stim_times)))) 

            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            plt.ylim(0,100)
            plt.show()

        ave_spikes_150.append(float(n_spikes_150)/float(len(stim_times)))

    if False:
        #Plot histogram from cumulative trains
        x = np.arange(0,10,.5)
        y = np.histogram(ave_spikes_150, bins = x)[0]
        plt.plot(x[:-1],np.float32(y), color='blue', linewidth=1)
        plt.show()

    return ave_spikes_150, ave_mean, cumulative_triggered_trains
    
    #Plot histograms
    bin_width = 0.010
    x = np.arange(0,0.500,bin_width)
    
    for i in range(len(Sort1.units)):
        y = np.histogram(cumulative_triggered_trains[i], bins = x)[0]
        plt.plot(x[:-1],np.float32(y)/len(stim_times), color='blue', linewidth=1.5)
        plt.title(Sort1.name + "\n unit: " + str(i) + " / " + str(len(Sort1.units))+ ";   no. spikes: " + str(len(Sort1.units[i])) +
        ";    firing rate over whole recording: " + str(float(len(Sort1.units[i]))/float(len(stim_times))) + " hz."
        "\n total no. of spikes in first 100ms: "+ str(sum(y[0:int(0.100/bin_width)])) +
        ";      ave no. of spikes in first 100ms: " + str(round(sum(y[0:int(0.100/bin_width)])/float(len(stim_times)),4)))
        
        plt.xlabel("Time from stimulus onset (seconds)")
        plt.ylabel("Binned firing rate (averaged over all stimuli; bin_width: "+str(bin_width))
        plt.ylim(0,2)
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.show()

            
def on_click(event):
    
    global coords, images_temp, ax, fig, cid
    
    n_pix = len(images_temp[0])
    
    if event.inaxes is not None:
        coords.append((event.ydata, event.xdata))
        for j in range(len(coords)):
            for k in range(3):
                for l in range(3):
                    images_temp[100][min(n_pix,int(coords[j][0])-1+k)][min(n_pix,int(coords[j][1])-1+l)]=0

        ax.imshow(images_temp[100])
        #plt.show()
        fig.canvas.draw()
                    #figManager = plt.get_current_fig_manager()
                    #figManager.window.showMaximized()
    else:
        print 'Exiting'
        plt.close()
        fig.canvas.mpl_disconnect(cid)


               
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

    genericmask_file = main_dir + '/roi_coords.txt'

    if (os.path.exists(genericmask_file)==False):
        fig, ax = plt.subplots()
        coords=[]

        ax.imshow(images_processed, vmin=-0.05, vmax=0.05)#, vmin=0.0, vmax=0.02)
        ax.set_title("Compute generic (outside the brain) mask")
        cid = fig.canvas.mpl_connect('button_press_event', on_click_roi)
        plt.show()

        np.savetxt(genericmask_file, coords)
        
        return np.int16(coords)
    else:
        
        return np.int16(np.loadtxt(genericmask_file))

        
def Define_generic_mask(images_processed, main_dir):

    global coords, images_temp, ax, fig, cid
    
    images_temp = images_processed.copy()


    if (os.path.exists(main_dir + '/genericmask.txt')==False):
        fig, ax = plt.subplots()
        coords=[]

        ax.imshow(images_processed[1000])#, vmin=0.0, vmax=0.02)
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
                    images_processed[100][i][j]=0
                    coords_save.append([i,j])
                counter+=1

        fig, ax = plt.subplots()
        ax.imshow(images_processed[100])
        plt.show()
       
        genericmask_file = main_dir + '/genericmask.txt'
        np.savetxt(genericmask_file, coords_save)

        print "Finished Making General Mask"

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
    
def Compress_LFP(lfp):

    file_name = lfp.name[:-4]+ '_scaled_notch.tsf'

    sub_sample_rate = 10

    out_SampleFrequency = lfp.SampleFrequency*1E+2/sub_sample_rate
    print "In samplefreq: ", lfp.SampleFrequency, "Out samplefreq: ", out_SampleFrequency

    start_electrodes = 0
    lfp.n_electrodes = 8 - start_electrodes

    lfp.n_vd_samples = lfp.n_vd_samples/2
    temp_data = []
    for i in range(start_electrodes, lfp.n_electrodes, 1):
        temp_data.append(lfp.lfp_traces[i][:lfp.n_vd_samples])
    lfp.lfp_traces=np.int16(temp_data)

    with open(file_name, 'wb') as f:
        f.write(lfp.header)
        f.write(struct.pack('i', lfp.iformat))
        f.write(struct.pack('i', out_SampleFrequency))
        f.write(struct.pack('i', lfp.n_electrodes))
        f.write(struct.pack('i', lfp.n_vd_samples/sub_sample_rate))
        f.write(struct.pack('f', lfp.vscale_HP))

        for i in range(start_electrodes, lfp.n_electrodes, 1):
            f.write(struct.pack('h', lfp.Siteloc[i*2]))
            f.write(struct.pack('h', lfp.Siteloc[i*2+1]))
            f.write(struct.pack('i', i+1))

        #Apply filters - if desired
        if True: 
            for i in range(start_electrodes, lfp.n_electrodes, 1):
                data = np.array(lfp.lfp_traces[i], dtype=np.int16)

                ##Apply butterworth filter
                #fs = lfp.SampleFrequency
                #lowcut = 50
                #highcut = 250.0
                #data = np.array(butter_bandpass_filter(data, lowcut, highcut, fs, order = 2), dtype=np.int16)

                #Apply 'notch" filter
                data = np.array(filter.notch(data)[0],dtype=np.int16) # remove 60 Hz mains noise, as for SI calc
                
                lfp.lfp_traces[i] = data

        #Remove spatial mean if desired
        if False: 
            data_mean = np.int16(np.mean(lfp.lfp_traces[:lfp.n_electrodes], 0))

        for i in range(start_electrodes, lfp.n_electrodes, 1):
            data = np.array(lfp.lfp_traces[i], dtype=np.int16)

            #(data[::sub_sample_rate]-data_mean[::sub_sample_rate]).tofile(f)
            data[::sub_sample_rate].tofile(f)

        f.write(struct.pack('i', 0)) #Write # of fake spikes
        f.close()   
    
    print "Exported notch filtered, temp compressed lfp: ", file_name

def Plot_histogram(Sort1):
    #Plot firing rate distribution of ground truth cells:
    if False:
       
        firing_rate_pyrs = []
        firing_rate_bcs = []
        for i in range(len(Sort1.units)):
            if len(Sort1.units[i])/240>1: #Exclude 1 spike units;  
                #if 'pyramid' in celldb[cell_index[i]][1]:
                    #print "pyr"
                    #for j in range(30):
                        firing_rate_pyrs.append(len(Sort1.units[i])/325)

                #if 'bc' in celldb[cell_index[i]][1]:
                #    print "bc"
                #    for j in range(30):
                #        firing_rate_bcs.append(len(Sort1.units[i])/325)

        font_size=45
        #Plot histogram from cumulative trains
        ax=plt.subplot(1,1,1)
        plt.ylabel("No. of cells", fontsize=font_size)
        plt.xlabel("Frequency (Hz)", fontsize=font_size)
        plt.xlim(0,15)
        plt.ylim(0,5)

        bin_width = 1.5
        x = np.arange(0,20,bin_width)
        y = np.histogram(firing_rate_pyrs, bins = x)
        plt.bar(y[1][:-1], y[0], bin_width-.1, color='blue', alpha=0.65)

        x = np.arange(0,20,bin_width)
        y = np.histogram(firing_rate_bcs, bins = x)
        plt.bar(y[1][:-1], y[0], bin_width-.1, color='red', alpha=0.65)

        labels = ['Excitatory', 'Inhibitory']
        excitatory = mpatches.Patch(facecolor = 'blue', edgecolor="black")
        inhibitory = mpatches.Patch(facecolor='red', edgecolor="black")
        
        first_legend = plt.legend([excitatory], labels, loc=1, prop={'size':25}) 

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.tick_params(axis='both', which='major', labelsize=font_size)
        plt.tick_params(axis='both', which='minor', labelsize=font_size)
        plt.show()    

        quit()


def Compute_darkneuron(Sort2):

    colors=['grey','blue','green','cyan','red', 'magenta',  'pink', 'brown']
    
    #Load electorde locs
    electrode_no = 0
    n_cells = 3198
    
    electrode = Electrode()
    probe_name = electrode.load(electrode_no)

    #Load saved ground truth cell ids for matching w. original data;
    cell_list = np.int32(np.genfromtxt('/media/cat/4TB/in_silico/ucsd_Sep20_rat_3k_20Khz/cell_index_0.csv'))

    #Find distance of ground truth cells to nearest electrode:
    #Load soma xyz locations and names from cell database
    f = open('/media/cat/4TB/in_silico/ucsd_Sep20_rat_3k_20Khz/rundir/3198cells/240000.0/input/celldb.csv', 'rt')
    reader = list(csv.reader(f))
    f.close()

    cell_distances = []
    for i in cell_list:
        soma_xyz=np.array([float(reader[i+1][2]),float(reader[i+1][3]),float(reader[i+1][4])])

        #Find closest electrode
        dist_array = electrode.probes[electrode_no]-[soma_xyz[0],-soma_xyz[2],soma_xyz[1]] #Compute distance between soma and all electrodes
        closest_electrode_dist = 10000.
        for j in range(len(dist_array)):
            cell_dist = np.sqrt(np.vdot(dist_array[j].T,dist_array[j])) #Shortest distance between cell and electorde
            if cell_dist < closest_electrode_dist:
                closest_electrode_dist = cell_dist

        cell_distances.append(closest_electrode_dist)
    
    sort_list = [ 'cat_sort2/', 'nicksw_sort2/', 'dan_sort2/','peter_sort2/', 'nickst_sort2/', 'pierre_sort2/', 'martin_sort2/']

    counter=0
    font_size=45
    for sort in sort_list:
        #Load list of units best match for each cell
        unit_match = np.int32(np.genfromtxt(Sort2.sim_dir + sort+ 'comparematch_silico_0_vs_ECP_0_allcells.csv'))

        #Find matching cells - and remove duplicate matches (oversplit units)
        matching_cell_list = cell_list[unit_match]
        matching_cell_list = np.unique(matching_cell_list)
        
        unit_distances = []
        for i in range(len(matching_cell_list)):
            unit_distances.append(cell_distances[np.where(cell_list == matching_cell_list[i])[0][0]])

        #Load allcells rasters
        file_name = '/media/cat/4TB/in_silico/ucsd_Sep20_rat_3k_20Khz/rasters.npy'
        rasters = np.load(file_name)*1E-3

        #Load allcells and find shortest distance to probe;
        #Skip cells that are < 20um from probe
        allcells_distances = []
        for i in range(n_cells):
            soma_xyz=np.array([float(reader[i+1][2]),float(reader[i+1][3]),float(reader[i+1][4])])
            #Find closest electrode
            dist_array = electrode.probes[electrode_no]-[soma_xyz[0],-soma_xyz[2],soma_xyz[1]] #Compute distance between soma and all electrodes
            closest_electrode_dist = 10000.
            for j in range(len(dist_array)):
                cell_dist = np.sqrt(np.vdot(dist_array[j].T,dist_array[j])) #Shortest distance between cell and electorde
                if cell_dist < closest_electrode_dist:
                    closest_electrode_dist = cell_dist
            if closest_electrode_dist < 15:
                pass
            else:
                #Exclude all cells firing < 0.5Hz, i.e. firing less than 120 spikes in 240 sec simulation
                if len(rasters[i]) > 120:  
                    allcells_distances.append(closest_electrode_dist)

        print "****DISTANCES FOR FILE: ", sort, "  ******"
        
        temp_array=[]
        x=[]
        bin_width = 5
        
        ##Plot concentric volume units/cells detected
        #for radius in range(20, 105, bin_width):
            #x.append(radius)
            #no_units = float(len(np.where(np.array(unit_distances)<radius)[0]))
            #no_cells = float(len(np.where(np.array(allcells_distances)<radius)[0]))
            #print "Radius: ", radius, "um  No. units: ", no_units, " % sorted: ", no_units/no_cells*100.
            #plt.plot([radius, radius], [0,100], 'r--', color='black', alpha=.5)
            #temp_array.append(no_units/no_cells*100.)
            
        #plt.plot(x,temp_array,color=colors[counter], linewidth=8, alpha=.8)
        #plt.ylabel("% Sorted cells sorted ", fontsize=font_size)
        #plt.xlabel("Maximum distance of cells to nearest electrode (um)", fontsize=font_size)
        #counter+=1

        ##Plot concentric shell units/cells detected
        for radius in range(20, 105, bin_width):
            #PLot # units and cells as a function of concentric shells
            no_units = len(np.where(np.logical_and(np.array(unit_distances)>=radius, np.array(unit_distances)<radius+bin_width))[0])
            no_cells = len(np.where(np.logical_and(np.array(cell_distances)>=radius, np.array(cell_distances)<radius+bin_width))[0])
            if no_cells>0:
                x.append(radius)
                temp_array.append(float(no_units)/float(no_cells)*100.)
            plt.plot([radius, radius], [0,100], 'r--', color='black', alpha=.5)
        plt.bar(np.array(x)+counter*.7, temp_array, .7, color=colors[counter], alpha=1)

        counter+=1

        #Plot concentric shell units/cells detected
        #for radius in range(20, 105, bin_width):
        #    #PLot # units and cells as a function of concentric shells
        #    no_units = len(np.where(np.logical_and(np.array(unit_distances)>=radius, np.array(unit_distances)<radius+bin_width))[0])
        #    no_cells = len(np.where(np.logical_and(np.array(cell_distances)>=radius, np.array(cell_distances)<radius+bin_width))[0])
        #    if no_cells>0:
        #        x.append(radius)
        #        temp_array.append(float(no_units)/float(no_cells)*100.)
        #    plt.plot([radius, radius], [0,100], 'r--', color='black', alpha=.5)
        #plt.bar(np.array(x)+counter*.7, temp_array, .7, color=colors[counter], alpha=1)

        #counter+=1
    
    plt.ylabel("% Sorted cells in concentric shell", fontsize=font_size)
    plt.xlabel("Radius of concentric shell (um)", fontsize=font_size)
    plt.xlim(20,100)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(axis='both', which='minor', labelsize=font_size)
    plt.show()
            

def Concatenate_ptcs(sim_dir, track_name, recordings):
    
    ptcs_flag = 1
    
    import re, glob

    #Make allptcs object
    Sorts = []
    rasters = []
    cumulative_time = 0
    for k in range(300): rasters.append([])
    for recording in recordings:
        temp_name = glob.glob(sim_dir+recording+"*")[0]
        temp_name = temp_name.replace('_lfp',''); temp_name = temp_name.replace('_mua','')
        file_name = temp_name.replace(sim_dir, '')
    
        work_dir = sim_dir + file_name + "/"
        Sort = Loadptcs(file_name, work_dir, ptcs_flag, save_timestamps=False)
        Sort.name=file_name
        Sort.filename=file_name
        Sort.directory=work_dir

        for k in range(len(Sort.units)):
            spikes = np.where(np.array(Sort.units[k])<1E10)[0] #REMOVE LARGE SPIKE TIMES THAT ARE BUGS FROM SS

        Sorts.append(Sort)

    return Sorts

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def Concatenate_lfp_events(sim_dir, recordings):
    
    import glob
    #Make allptcs object
    Sorts = []
    rasters = []
    cumulative_time = 0

    for recording in recordings:
        file_name = glob.glob(sim_dir+recording+"*")[0]
        file_name = file_name.replace(sim_dir, '')

        units = np.load(sim_dir+file_name+'/'+file_name+'.lfp.events.npy')
        
        Sorts.append(units)

    return Sorts


def Split_lfp_ptcs(Sort_lfp, lfp, recordings):
    ''' Function converts track wide lfp sorted events into individual event files for each recording directory. As
    part of processing the lfp compressed event times are converted back to ms (which is the sampling rate of originl lfp data).
    '''
    import glob

    #Convert lfp spiketimes from compressed/subsampled format back to real time divide 200 to get KHZ samplerate
    temp_units = []
    for k in range(len(Sort_lfp.units)):
        #temp_units = np.array(Sort_lfp.units[k])/100.*1E3    #10-subsample; Converted to seconds and then to ms  
        temp_units = np.array(Sort_lfp.units[k])/200.*1E3    #5-subsample; Converted to seconds and then to ms  
        #temp_units = np.array(Sort_lfp.units[k])/1000.*1E3    #1-subsample; Converted to seconds and then to ms  
        Sort_lfp.units[k] = temp_units

    temp_spikes = 0
    for i in range(len(recordings)):
        #Find correct file name to save lfp event data
        file_name = glob.glob(Sort_lfp.directory+recordings[i]+"*")[0]
        file_name = file_name.replace(Sort_lfp.directory, '')
        
        units = []  #Hold list of matching spike for each recording
        start = sum(lfp.t_end[:i])*1E3
        end = start + lfp.t_end[i]*1E3
        for u in range(len(Sort_lfp.units)):
            temp_indexes = np.where(np.logical_and(Sort_lfp.units[u]>=start, Sort_lfp.units[u]<=end))[0]
            units.append(np.array(Sort_lfp.units[u])[temp_indexes]-start)
            
        np.save(Sort_lfp.directory+file_name+'/'+file_name+'.lfp.events', units)

def Save_lfp_ptcs(Sort_lfp, lfp, recordings, track_name):
    ''' Function converts single compressed lfp .ptcs file to a single ".lfp.events" file
    '''
    import glob

    #Convert lfp spiketimes from compressed/subsampled format back to real time divide 200 to get KHZ samplerate
    temp_units = []
    for k in range(len(Sort_lfp.units)):
        if '12-1-1' in track_name:
            temp_units = np.array(Sort_lfp.units[k])#/100.*1E3  #Converted to seconds and then to ms; NB: ONLY TIM"S DATA!!!!!!
        else:
            temp_units = np.array(Sort_lfp.units[k])/100.*1E3  #Converted to seconds and then to ms; NB: ONLY TIM"S DATA!!!!!!
        Sort_lfp.units[k] = temp_units

    temp_spikes = 0
    #Find correct file name to save lfp event data
    file_name = glob.glob(Sort_lfp.directory+recordings[0]+"*")[0]
    file_name = file_name.replace(Sort_lfp.directory, '')
    
    units = []  #Hold list of matching spike for each recording
    start = sum(lfp.t_end[0])*1E3
    end = start + lfp.t_end[0]*1E3
    for u in range(len(Sort_lfp.units)):
        print "lfp unit: ", u
        units.append(np.array(Sort_lfp.units[u]))
        
    np.save(Sort_lfp.directory+file_name+'/'+file_name+'.lfp.events', units)

#****************************************************************

def MCD_read_imagingtimes_old(MCDFilePath):
 
    #import necessary libraries
 
    import neuroshare as ns
    import numpy as np
 
    #open file using the neuroshare bindings
    fd = ns.File(MCDFilePath)
 
    #create index
    indx = 0
 
    #create empty dictionary
    data = dict()
 
    #loop through data and find all analog entities
 
    for entity in fd.list_entities():
        #print "looping over entities: ", entity

        if entity.entity_type == 1:
            data["extra"] = fd.entities[indx]
    return data


def MCD_read_imagingtimes(MCDFilePath):
 
    #import necessary libraries
 
    import neuroshare as ns
    import numpy as np
 
    #open file using the neuroshare bindings
    fd = ns.File(MCDFilePath)
 
    #create index
    indx = 0

    #loop through data and find all analog entities

    counter=0
    data=[]
    for entity in fd.list_entities():
        #print "looping over entities: ", entity

        if entity.entity_type == 1:
            data.append(fd.entities[indx])
            print fd.entities[indx]

    return data


def synchrony_index(data, lfp, rec_index, si_limit):

    """Calculate an LFP synchrony index, using potentially overlapping windows of width
    and tres, in sec, from the LFP spectrogram, itself composed of bins of lfpwidth and
    lfptres. Note that for power ratio methods (kind: L/(L+H) or L/H), width and tres are
    not used, only lfpwidth and lfptres. Options for kind are:
    'L/(L+H)': fraction of power in low band vs total power (Saleem2010)
    'L/H': low to highband power ratio (Li, Poo, Dan 2009)
    'cv': coefficient of variation (std / mean) of all power
    'ncv': normalized CV: (std - mean) / (std + mean)
    'nstdmed': normalized stdmed: (std - med) / (std + med)
    'n2stdmean': normalized 2stdmean: (2*std - mean) / (2*std + mean)
    'n3stdmean': normalized 3stdmean: (3*std - mean) / (3*std + mean)
    relative2t0 controls whether to plot relative to t0, or relative to start of ADC
    clock. lim2stim limits the time range only to when a stimulus was presented, i.e. to
    the outermost times of non-NULL din.
    """

    ts = np.arange(0,len(data),1E3)   #set of timestamps, in sec
    t0i, t1i = int(ts[0]), int(ts[-1])
    t0 = t0i
    t1 = t1i
    
    x = data[t0i:t1i] / 1e3 # slice data, convert from uV to mV
    x = filter.notch(x)[0] # remove 60 Hz mains noise

    lfpwidth=30
    lfptres=5     #Time resolution: bin width for analysis in seconds
    
    lowband = [0.1, 7]; highband = [15,100]     #Martin's suggestions; Saleem 2010?
    #lowband = [0.1, 4]; highband = [15,100]     #variations
    #lowband = [0.1, 4]; highband = [20,100]     #variations

    f0, f1 = lowband
    f2, f3 = highband

    assert lfptres <= lfpwidth
    NFFT = intround(lfpwidth * lfp.sampfreq)
    noverlap = intround(NFFT - lfptres * lfp.sampfreq)
    # t is midpoints of timebins in sec from start of data. P is in mV^2?:
    P, freqs, t = mpl.mlab.specgram(x, NFFT=NFFT, Fs=lfp.sampfreq, noverlap=noverlap)

    # keep only freqs between f0 and f1, and f2 and f3:
    f0i, f1i, f2i, f3i = freqs.searchsorted([f0, f1, f2, f3])
    lP = P[f0i:f1i] # nsubfreqs x nt
    hP = P[f2i:f3i] # nsubfreqs x nt
    lP = lP.sum(axis=0) # nt
    hP = hP.sum(axis=0) # nt
    
    # calculate some metric of each column, ie each width:
    kind = 'L/(L+H)'
    si = lP/(lP + hP) #Saleem 2010

    #Insert initial time point into si index
    si = np.insert(si,0,si[0])
    si = np.insert(si,len(si),si[-1])
    t = np.insert(t,0,lfp.t_start[rec_index])
    t = np.insert(t,len(t),lfp.t_end[rec_index])

    #**** Find periods of synchrony > si_limit = 0.7:
    #si_limit = 0.7
    sync_periods = []
    in_out = 0
    for i in range(len(t)):
        if si[i]>si_limit:
            if in_out==0:
                ind1=t[i]
                in_out=1
        else:
            if in_out==1:
                ind2=t[i-1]
                sync_periods.append([ind1,ind2])
                in_out=0
    if in_out==1:
        sync_periods.append([ind1,t[-1]])

    return si, t, sync_periods # t are midpoints of bins, from start of ADC clock


def synchrony_index_martin(self, kind=None, chani=-1, width=None, tres=None,
       lfpwidth=None, lfptres=None, lowband=None, highband=None, plot=True,
       states=False, desynchsi=0.2, synchsi=0.2, lw=4, alpha=1, relative2t0=False,
       lim2stim=False, showxlabel=True, showylabel=True, showtitle=True, showtext=True,
       swapaxes=False, figsize=(20, 3.5)):
    """Calculate an LFP synchrony index, using potentially overlapping windows of width
    and tres, in sec, from the LFP spectrogram, itself composed of bins of lfpwidth and
    lfptres. Note that for power ratio methods (kind: L/(L+H) or L/H), width and tres are
    not used, only lfpwidth and lfptres. Options for kind are:
    'L/(L+H)': fraction of power in low band vs total power (Saleem2010)
    'L/H': low to highband power ratio (Li, Poo, Dan 2009)
    'cv': coefficient of variation (std / mean) of all power
    'ncv': normalized CV: (std - mean) / (std + mean)
    'nstdmed': normalized stdmed: (std - med) / (std + med)
    'n2stdmean': normalized 2stdmean: (2*std - mean) / (2*std + mean)
    'n3stdmean': normalized 3stdmean: (3*std - mean) / (3*std + mean)
    relative2t0 controls whether to plot relative to t0, or relative to start of ADC
    clock. lim2stim limits the time range only to when a stimulus was presented, i.e. to
    the outermost times of non-NULL din.
    """
    uns = get_ipython().user_ns
    if kind == None:
        kind = uns['LFPSIKIND']
    if kind.startswith('L/'):
        pratio = True
    else:
        pratio = False

    data = self.get_data()
    ts = self.get_tssec() # full set of timestamps, in sec
    t0, t1 = ts[0], ts[-1]
    if lim2stim:
        t0, t1 = self.apply_lim2stim(t0, t1)
    t0i, t1i = ts.searchsorted((t0, t1))
    x = data[chani, t0i:t1i] / 1e3 # slice data, convert from uV to mV
    x = filter.notch(x)[0] # remove 60 Hz mains noise
    try:
        rr = self.r.e0.I['REFRESHRATE']
    except AttributeError: # probably a recording with no experiment
        rr = 200 # assume 200 Hz refresh rate
    if rr <= 100: # CRT was at low vertical refresh rate
        print('filtering out %d Hz from LFP in %s' % (intround(rr), self.r.name))
        x = filter.notch(x, freq=rr)[0] # remove CRT interference

    if width == None:
        width = uns['LFPSIWIDTH'] # sec
    if tres == None:
        tres = uns['LFPSITRES'] # sec
    if lfpwidth == None:
        lfpwidth = uns['LFPWIDTH'] # sec
    if lfptres == None:
        lfptres = uns['LFPTRES'] # sec
    if lowband == None:
        lowband = uns['LFPSILOWBAND']
    f0, f1 = lowband
    if highband == None:
        highband = uns['LFPSIHIGHBAND']
    f2, f3 = highband

    assert lfptres <= lfpwidth
    NFFT = intround(lfpwidth * self.sampfreq)
    noverlap = intround(NFFT - lfptres * self.sampfreq)
    #print('len(x), NFFT, noverlap: %d, %d, %d' % (len(x), NFFT, noverlap))
    # t is midpoints of timebins in sec from start of data. P is in mV^2?:
    P, freqs, Pt = mpl.mlab.specgram(x, NFFT=NFFT, Fs=self.sampfreq, noverlap=noverlap)
    # don't convert power to dB, just washes out the signal in the ratio:
    #P = 10. * np.log10(P)
    if not relative2t0:
        Pt += t0 # convert t to time from start of ADC clock:
    nfreqs = len(freqs)

    # keep only freqs between f0 and f1, and f2 and f3:
    f0i, f1i, f2i, f3i = freqs.searchsorted([f0, f1, f2, f3])
    lP = P[f0i:f1i] # nsubfreqs x nt
    hP = P[f2i:f3i] # nsubfreqs x nt
    lP = lP.sum(axis=0) # nt
    hP = hP.sum(axis=0) # nt
    
    #si = lP/(lP + hP)

    if pratio:
        t = Pt
        ylim = 0, 1
        ylabel = 'SI (%s)' % kind
    else:
        # potentially overlapping bin time ranges:
        trange = Pt[0], Pt[-1]
        tranges = split_tranges([trange], width, tres) # in sec
        ntranges = len(tranges)
        tis = Pt.searchsorted(tranges) # ntranges x 2 array
        # number of timepoints to use for each trange, almost all will be the same width:
        binnt = intround((tis[:, 1] - tis[:, 0]).mean())
        binhP = np.zeros((ntranges, binnt)) # init appropriate array
        for trangei, t0i in enumerate(tis[:, 0]):
            binhP[trangei] = hP[t0i:t0i+binnt]
        # get midpoint of each trange:
        t = tranges.mean(axis=1)

    #old_settings = np.seterr(all='ignore') # suppress div by 0 errors
    # plot power signal to be analyzed
    #self.si_plot(Pt, hP, t0=0, t1=t[-1], ylim=None, ylabel='highband power',
    #             title=lastcmd()+' highband power', text=self.r.name)
    hlines = []
    if kind[0] == 'n':
        ylim = -1, 1
        hlines = [0]
    # calculate some metric of each column, ie each width:
    if kind == 'L/(L+H)':
        si = lP/(lP + hP)
    elif kind == 'L/H':
        si = lP/hP
    elif kind == 'nLH':
        t = Pt
        si = (lP - hP) / (lP + hP)
        ylabel = 'LFP (L - H) / (L + H)'
    elif kind == 'cv':
        si = binhP.std(axis=1) / binhP.mean(axis=1)
        ylim = 0, 2
        ylabel = 'LFP power CV'
    elif kind == 'ncv':
        s = binhP.std(axis=1)
        mean = binhP.mean(axis=1)
        si = (s - mean) / (s + mean)
        ylabel = 'LFP power (std - mean) / (std + mean)'
        #pl.plot(t, s)
        #pl.plot(t, mean)
    elif kind == 'n2stdmean':
        s2 = 2 * binhP.std(axis=1)
        mean = binhP.mean(axis=1)
        si = (s2 - mean) / (s2 + mean)
        ylabel = 'LFP power (2*std - mean) / (2*std + mean)'
        hlines = [-0.1, 0, 0.1] # demarcate desynched and synched thresholds
        #pl.plot(t, s2)
        #pl.plot(t, mean)
    elif kind == 'n3stdmean':
        s3 = 3 * binhP.std(axis=1)
        mean = binhP.mean(axis=1)
        si = (s3 - mean) / (s3 + mean)
        ylabel = 'LFP power (3*std - mean) / (3*std + mean)'
        hlines = [-0.1, 0, 0.1] # demarcate desynched and synched thresholds
        #pl.plot(t, s3)
        #pl.plot(t, mean)
    elif kind == 'n4stdmean':
        s4 = 4 * binhP.std(axis=1)
        mean = binhP.mean(axis=1)
        si = (s4 - mean) / (s4 + mean)
        ylabel = 'LFP power (4*std - mean) / (4*std + mean)'
        #pl.plot(t, s4)
        #pl.plot(t, mean)
    elif kind == 'nstdmed':
        s = binhP.std(axis=1)
        med = np.median(binhP, axis=1)
        si = (s - med) / (s + med)
        ylabel = 'LFP power (std - med) / (std + med)'
        hlines = [-0.1, 0, 0.1] # demarcate desynched and synched thresholds
        #pl.plot(t, s)
        #pl.plot(t, med)
    elif kind == 'n2stdmed':
        s2 = 2 * binhP.std(axis=1)
        med = np.median(binhP, axis=1)
        si = (s2 - med) / (s2 + med)
        ylabel = 'LFP power (2*std - med) / (2*std + med)'
        hlines = [-0.1, 0, 0.1] # demarcate desynched and synched thresholds
        #pl.plot(t, s2)
        #pl.plot(t, med)
    elif kind == 'n3stdmed':
        s3 = 3 * binhP.std(axis=1)
        med = np.median(binhP, axis=1)
        si = (s3 - med) / (s3 + med)
        ylabel = 'LFP power (3*std - med) / (3*std + med)'
        hlines = [-0.1, 0, 0.1] # demarcate desynched and synched thresholds
        #pl.plot(t, s3)
        #pl.plot(t, med)
    elif kind == 'nstdmin':
        s = binhP.std(axis=1)
        min = binhP.min(axis=1)
        si = (s - min) / (s + min)
        ylabel = 'LFP power (std - min) / (std + min)'
        #pl.plot(t, s)
        #pl.plot(t, min)
    elif kind == 'nmadmean':
        mean = binhP.mean(axis=1)
        mad = (np.abs(binhP - mean[:, None])).mean(axis=1)
        si = (mad - mean) / (mad + mean)
        ylabel = 'MUA (MAD - mean) / (MAD + mean)'
        #pl.plot(t, mad)
        #pl.plot(t, mean)
    elif kind == 'nmadmed':
        med = np.median(binhP, axis=1)
        mad = (np.abs(binhP - med[:, None])).mean(axis=1)
        si = (mad - med) / (mad + med)
        ylabel = 'MUA (MAD - median) / (MAD + median)'
        #pl.plot(t, mad)
        #pl.plot(t, med)
    elif kind == 'nvarmin':
        v = binhP.var(axis=1)
        min = binhP.min(axis=1)
        si = (v - min) / (v + min)
        ylabel = 'LFP power (std - min) / (std + min)'
        #pl.plot(t, v)
        #pl.plot(t, min)
    elif kind == 'nptpmean':
        ptp = binhP.ptp(axis=1)
        mean = binhP.mean(axis=1)
        si = (ptp - mean) / (ptp + mean)
        ylabel = 'MUA (ptp - mean) / (ptp + mean)'
        #pl.plot(t, ptp)
        #pl.plot(t, mean)
    elif kind == 'nptpmed':
        ptp = binhP.ptp(axis=1)
        med = np.median(binhP, axis=1)
        si = (ptp - med) / (ptp + med)
        ylabel = 'MUA (ptp - med) / (ptp + med)'
        #pl.plot(t, ptp)
        #pl.plot(t, med)
    elif kind == 'nptpmin':
        ptp = binhP.ptp(axis=1)
        min = binhP.min(axis=1)
        si = (ptp - min) / (ptp + min)
        ylabel = 'MUA (ptp - min) / (ptp + min)'
        #pl.plot(t, ptp)
        #pl.plot(t, min)
    elif kind == 'nmaxmin':
        max = binhP.max(axis=1)
        min = binhP.min(axis=1)
        si = (max - min) / (max + min)
        ylabel = 'MUA (max - min) / (max + min)'
        #pl.plot(t, max)
        #pl.plot(t, min)
    else:
        raise ValueError('unknown kind %r' % kind)
    if plot:
        # calculate xlim, always start from 0, add half a bin width to xmax:
        if pratio:
            xlim = (0, t[-1]+lfpwidth/2)
        else:
            xlim = (0, t[-1]+width/2)
        self.si_plot(t, si, t0=t0, t1=t1, xlim=xlim, ylim=ylim, ylabel=ylabel,
                     showxlabel=showxlabel, showylabel=showylabel, showtitle=showtitle,
                     title=lastcmd(), showtext=showtext, text=self.r.name, hlines=hlines,
                     states=states, desynchsi=desynchsi, synchsi=synchsi, lw=lw,
                     alpha=alpha, relative2t0=relative2t0, swapaxes=swapaxes,
                     figsize=figsize)
    #np.seterr(**old_settings) # restore old settings
    return si, t # t are midpoints of bins, from start of ADC clock

def xcorr_maxlag(x, y, maxlag=1000.0):
    """
    Compute the cross-correlogram of two time series.
    """
    xl = x.size
    yl = y.size

    c = np.zeros(2*maxlag + 1)

    for i in xrange(int(maxlag)+1):
        tmp = np.corrcoef(x[0:min(xl, yl-i)], y[i:i+min(xl, yl-i)])
        c[maxlag-i] = tmp[1][0]
        tmp = np.corrcoef(x[i:i+min(xl-i, yl)], y[0:min(xl-i, yl)])
        c[maxlag+i] = tmp[1][0]

    return c

def MUA_compute(Sort, bin_width):
    
    print "Computing MUA spikes for bin_width: ", bin_width
    mua_spikes = []
    for i in range(len(Sort.units)):
        mua_spikes.extend(Sort.units[i])
    mua_spikes = np.array(mua_spikes)/Sort.samplerate
    x = np.arange(0,max(np.array(mua_spikes)),bin_width)
    y = np.histogram(mua_spikes, bins = x)
    return x[:-1], y[0]
    
    #print "Plotting MUA"
    #plt.plot(x[:-1], y[0], color='red', linewidth=2, alpha=0.65)
    #plt.show()
    #quit()

class Object_temp(object):
    pass

    
def PCA(X, n_components):
    from sklearn import decomposition
    
    print "...decomposition..."
    pca = decomposition.PCA(n_components)

    print "...fit..."
    pca.fit(X)
        
    print "...transform..."
    X=pca.transform(X)

    coords = []
    for i in range(len(X)):
         coords.append([X[i][0], X[i][1], X[i][2]])

    
    return X, np.array(coords).T

def KMEANS(data, n_clusters):

    from sklearn import cluster, datasets
    clusters = cluster.KMeans(n_clusters, max_iter=1000, n_jobs=-1, random_state = 1032)
    clusters.fit(data)
    
    return clusters.labels_


def intround(n):
    """Round to the nearest integer, return an integer. Works on arrays.
    Saves on parentheses, nothing more"""
    if iterable(n): # it's a sequence, return as an int64 array
        return np.int64(np.round(n))
    else: # it's a scalar, return as normal Python int
        return int(round(n))
    
   
