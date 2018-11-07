
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import IPython.display
from ipywidgets import interact, interactive, fixed
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import scipy.ndimage
import socket
import os
import sys
import pandas as pd
import scipy.io as sio
import matplotlib
import scipy.signal as sg
import math
import scipy as sp
import pylab
import os
import wave
import scipy.io.wavfile
import h5py
import pickle
import tensorflow as tf
import random
import h5py
import collections
import bisect
import platform
import random


# In[2]:

#date and bird of data used
bird_id = 'z007'
num_clusters = 64
model_name = 'spectrogram prediction model/mel'

#locate source folder
hostname = socket.gethostname()
    
if hostname == 'txori' or hostname == 'passaro':
    data_folder = os.path.abspath('/mnt/cube/kai/data')
    results_folder = os.path.abspath('/mnt/cube/kai/results/'+model_name)
    repos_folder = os.path.abspath('/mnt/cube/kai/repositories')
    data_folder_zeke = os.path.abspath('/mnt/cube/earneodo/bci_zf/ss_data')
    bird_folder_save = os.path.join(data_folder, bird_id)
    bird_folder_data = os.path.join(data_folder_zeke, bird_id)
else:
    data_folder = os.path.abspath('/Users/Kai/Documents/UCSD/Research/Gentner group/Raw Data')
    results_folder = os.path.abspath('/Users/Kai/Documents/UCSD/Research/Gentner group/Results/'+model_name)
    repos_folder = os.path.abspath('/Users/Kai/Documents/UCSD/Research/Gentner group/Code')
    bird_folder_save = os.path.join(data_folder, bird_id)
    bird_folder_data = bird_folder_save
    
if not os.path.exists(bird_folder_save):
    os.makedirs(bird_folder_save)

if not os.path.exists(results_folder):
    os.makedirs(results_folder)


# In[3]:

# Most of the Spectrograms and Inversion are taken from: https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    window_step : int
        Step size between windows
    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))

    ws = window_size
    ss = window_step
    a = X

    valid = len(a) - ws
    nw = int((valid) // ss)
    out = np.ndarray((nw,ws),dtype = a.dtype)

    for i in xrange(nw):
        # "slide" the window along the samples
        start = i * ss
        stop = start + ws
        out[i] = a[start : stop]

    return out


def stft(X, fftsize=128, step=65, mean_normalize=True, real=False,
         compute_onesided=True):
    """
    Compute STFT for 1D real valued input X
    """
    if real:
        local_fft = np.fft.rfft
        cut = -1
    else:
        local_fft = np.fft.fft
        cut = None
    if compute_onesided:
        cut = fftsize // 2
    if mean_normalize:
        X -= X.mean()

    X = overlap(X, fftsize, step)
    
    size = fftsize
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    X = X * win[None]
    X = local_fft(X)[:, :cut]
    return X

def pretty_spectrogram(d,log = True, thresh= 5, fft_size = 512, step_size = 64):
    """
    creates a spectrogram
    log: take the log of the spectrgram
    thresh: threshold minimum power for log spectrogram
    """
    specgram = np.abs(stft(d, fftsize=fft_size, step=step_size, real=False,
        compute_onesided=True))
  
    if log == True:
        specgram /= specgram.max() # volume normalize to max 1
        specgram = np.log10(specgram) # take log
        specgram[specgram < -thresh] = -thresh # set anything less than the threshold as the threshold
    else:
        specgram[specgram < thresh] = thresh # set anything less than the threshold as the threshold
    
    return specgram

# Also mostly modified or taken from https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
def invert_pretty_spectrogram(X_s, log = True, fft_size = 512, step_size = 512/4, n_iter = 10):
    
    if log == True:
        X_s = np.power(10, X_s)

    X_s = np.concatenate([X_s, X_s[:, ::-1]], axis=1)
    X_t = iterate_invert_spectrogram(X_s, fft_size, step_size, n_iter=n_iter)
    return X_t

def iterate_invert_spectrogram(X_s, fftsize, step, n_iter=10, verbose=False):
    """
    Under MSR-LA License
    Based on MATLAB implementation from Spectrogram Inversion Toolbox
    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.
    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.
    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    reg = np.max(X_s) / 1E8
    X_best = copy.deepcopy(X_s)
    for i in range(n_iter):
        if verbose:
            print("Runnning iter %i" % i)
        if i == 0:
            X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                                     set_zero_phase=True)
        else:
            # Calculate offset was False in the MATLAB version
            # but in mine it massively improves the result
            # Possible bug in my impl?
            X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                                     set_zero_phase=False)
        est = stft(X_t, fftsize=fftsize, step=step, compute_onesided=False)
        phase = est / np.maximum(reg, np.abs(est))
        X_best = X_s * phase[:len(X_s)]
    X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                             set_zero_phase=False)
    return np.real(X_t)

def invert_spectrogram(X_s, step, calculate_offset=True, set_zero_phase=True):
    """
    Under MSR-LA License
    Based on MATLAB implementation from Spectrogram Inversion Toolbox
    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.
    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.
    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    size = int(X_s.shape[1] // 2)
    wave = np.zeros((X_s.shape[0] * step + size))
    # Getting overflow warnings with 32 bit...
    wave = wave.astype('float64')
    total_windowing_sum = np.zeros((X_s.shape[0] * step + size))
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))

    est_start = int(size // 2) - 1
    est_end = est_start + size
    for i in range(X_s.shape[0]):
        wave_start = int(step * i)
        wave_end = wave_start + size
        if set_zero_phase:
            spectral_slice = X_s[i].real + 0j
        else:
            # already complex
            spectral_slice = X_s[i]

        # Don't need fftshift due to different impl.
        wave_est = np.real(np.fft.ifft(spectral_slice))[::-1]
        if calculate_offset and i > 0:
            offset_size = size - step
            if offset_size <= 0:
                print("WARNING: Large step size >50\% detected! "
                      "This code works best with high overlap - try "
                      "with 75% or greater")
                offset_size = step
            offset = xcorr_offset(wave[wave_start:wave_start + offset_size],
                                  wave_est[est_start:est_start + offset_size])
        else:
            offset = 0
        wave[wave_start:wave_end] += win * wave_est[
            est_start - offset:est_end - offset]
        total_windowing_sum[wave_start:wave_end] += win
    wave = np.real(wave) / (total_windowing_sum + 1E-6)
    return wave

def xcorr_offset(x1, x2):
    """
    Under MSR-LA License
    Based on MATLAB implementation from Spectrogram Inversion Toolbox
    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.
    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.
    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()
    frame_size = len(x2)
    half = frame_size // 2
    corrs = np.convolve(x1.astype('float32'), x2[::-1].astype('float32'))
    corrs[:half] = -1E30
    corrs[-half:] = -1E30
    offset = corrs.argmax() - len(x1)
    return offset

def make_mel(spectrogram, mel_filter, shorten_factor = 1):
    mel_spec =np.transpose(mel_filter).dot(np.transpose(spectrogram))
    mel_spec = scipy.ndimage.zoom(mel_spec.astype('float32'), [1, 1./shorten_factor]).astype('float16')
    mel_spec = mel_spec[:,1:-1] # a little hacky but seemingly needed for clipping 
    return mel_spec

def mel_to_spectrogram(mel_spec, mel_inversion_filter, spec_thresh, shorten_factor):
    """
    takes in an mel spectrogram and returns a normal spectrogram for inversion 
    """
    mel_spec = (mel_spec+spec_thresh)
    uncompressed_spec = np.transpose(np.transpose(mel_spec).dot(mel_inversion_filter))
    uncompressed_spec = scipy.ndimage.zoom(uncompressed_spec.astype('float32'), [1,shorten_factor]).astype('float16')
    uncompressed_spec = uncompressed_spec -4
    return uncompressed_spec

# From https://github.com/jameslyons/python_speech_features

def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.)
    
def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(nfilt=20,nfft=512,samplerate=30000,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.

    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"
    
    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = np.zeros([nfilt,nfft//2])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        if int(bin[j+1]) == int(bin[j+2]):
            fbank[j,int(bin[j+1])] = (int(bin[j+1]) - bin[j]) / (bin[j+1]-bin[j])
        else:
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank

def create_mel_filter(fft_size, n_freq_components = 64, start_freq = 300, end_freq = 8000, samplerate=30000):
    """
    Creates a filter to convolve with the spectrogram to get out mels

    """
    mel_inversion_filter = get_filterbanks(nfilt=n_freq_components, 
                                           nfft=fft_size, samplerate=samplerate, 
                                           lowfreq=start_freq, highfreq=end_freq)
    # Normalize filter
    mel_filter = mel_inversion_filter.T / mel_inversion_filter.sum(axis=1)

    return mel_filter, mel_inversion_filter


# In[4]:

def make_datasets_spec(num_lookback, flatten=True, spec_file = 'mel_spectrogram.dat'):
    
    spec_file = os.path.join(bird_folder, spec_name)
    spec = np.loadtxt(spec_file).tolist()
        
    subdirs = [combined[0] for combined in os.walk(bird_folder)][1:]
    neural_datasets = list()
    all_datasets = list()

    for directory in subdirs:
        pickle_file = os.path.join(directory, pickle_name_mel)
        '''
        neuraldata = pickle.load(open(pickle_file, "rb" ))
        '''
        try:
            neuraldata = pickle.load(open(pickle_file, "rb" ))
        except IOError:
            continue
        
        num_bins = len(neuraldata[0])
        #print(neuraldata[0][0])
        for i in range(len(neuraldata)):
            for _ in range(num_lookback):
                neuraldata[i].insert(0, np.zeros(len(neuraldata[i][0])))
            
            for j in range(num_bins):
                if flatten:
                    neuralvector = np.concatenate(neuraldata[i][j:j+num_lookback])
                else:
                    neuralvector = np.array(neuraldata[i][j:j+num_lookback])
                neural_datasets.append(neuralvector)
                #for k in range(num_lookback):
               
                    #neuralvector.extend(neuraldata[i][j+k])
        
            zipped = zip(neural_datasets, spec)
            all_datasets = all_datasets+zipped
    return all_datasets


# In[5]:

def spike_sort(neural_kwik_name, song_kwik_name, song_length, bin_size, num_clusters, num_lookbacks, start_extend_bins=0, 
               end_extend_bins=0, bird_folder_save=bird_folder_save, bird_folder_data=bird_folder_data, specify_subdir=None,
              n_mel_freq_components = 64, start_freq = 200, end_freq = 15000):
    '''
    neural_kwik_name: kwik file that contains neural data
    song_kwik_name: kwik file that contains motif starting time and recordings
    song_length: length of motif
    bin_size: how many datapoints to combine into a bin
    num_clusters: how many clusters of neurons are recorded
    start_extend_bins: at the beginning of the motif, how many bins do you want to shift. Shifting the start to the left
                        is negative, and shifting to the right is positive. Default to be 0
    end_extend_bins: same as start_extend_bins, how many bins to shift at the end of motif. Shifting left is negative and
                        right is positive. Default to be 0.
    
    '''
    save_name = 'mel_%02d_%d_%d.p' %(num_lookbacks, start_extend_bins, end_extend_bins)
    
    fft_size = bin_size*16 # window size for the FFT
    step_size = bin_size # distance to slide along the window (in time)
    start_extend_bins *= int(fft_size/bin_size)
    end_extend_bins *= int(fft_size/bin_size)
    spec_thresh = 5 # threshold for spectrograms (lower filters out more noise)
    lowcut = 500 # Hz # Low cut for our butter bandpass filter
    highcut = 15000 # Hz # High cut for our butter bandpass filter
    # For mels
    # number of mel frequency channels
    shorten_factor = 1 # how much should we compress the x-axis (time)
     # Hz # What frequency to start sampling our melS from 
     # Hz # What frequency to stop sampling our melS from 
    
    mel_filter, mel_inversion_filter = create_mel_filter(fft_size = fft_size,
                                                        n_freq_components = n_mel_freq_components,
                                                        start_freq = start_freq,
                                                        end_freq = end_freq)
    
    num_bins = int(song_length/bin_size-2-start_extend_bins+end_extend_bins)#number of bins in each song
    
    if specify_subdir:
        subdirs = [os.path.join(bird_folder_data, specify_subdir)]
    else:
        subdirs = [combined[0] for combined in os.walk(bird_folder_data)][1:]
        
    extended_song_length = song_length+bin_size*(end_extend_bins-start_extend_bins)
    neural_song_length = extended_song_length+(num_lookbacks-2)*bin_size 
    #take out two bins due to mel processing algorithm
    
    mel_neural_comb = list()
    tot_motifs = 0
    
    for directory in subdirs:
        kwik_file = os.path.join(directory, neural_kwik_name)
        song_file = os.path.join(directory, song_kwik_name)
        
        try:
            trial1 = h5py.File(kwik_file,'r')['/channel_groups/0/spikes/time_samples'].value
        except IOError:
            print('--->Skipping '+directory+' because it doesnt contain spiking kwik files.')
            continue
            
        if "sleep" in directory:
            print('--->Skipping '+directory+' because it is a sleep recording.')
            continue
            
        recording = directory[directory.index('day-'):]
        recording_folder_save = os.path.join(bird_folder_save, recording)
        if not os.path.exists(recording_folder_save):
            os.makedirs(recording_folder_save)
            
        print('Start working on '+directory)
        
        with h5py.File(kwik_file,'r') as f:
            time_samples = f['/channel_groups/0/spikes/time_samples'].value
            cluster_num = f['/channel_groups/0/spikes/clusters/main'].value
            recording_num = f['/channel_groups/0/spikes/recording'].value
            if max(recording_num)>20:
                error_indices = [i for i,v in enumerate(list(recording_num)) if v > 20]
                for index in error_indices:
                    recording_num[index] = recording_num[index-1]
                print('***Erroneous indices fixed.')
                
            #print(type(time_samples))
            rec_counter = collections.Counter(recording_num)
            rec_list = rec_counter.keys()
            spike_counts = rec_counter.values()
            time_samples_each_rec = list()
            cluster_each_rec = list()

            #separate time samples based on recording
            for rec in rec_list:
                #start_samples.append(f['/recordings/'][str(rec)].attrs['start_sample'])
                prev_num_samples = sum(spike_counts[:rec])
                time_samples_each_rec.append(time_samples[prev_num_samples:prev_num_samples+spike_counts[rec]])
                #time_samples[prev_num_samples:prev_num_samples+spike_counts[rec]] = time_samples[
                #    prev_num_samples:prev_num_samples+spike_counts[rec]]+f['/recordings/'][str(rec)].attrs['start_sample']
                cluster_each_rec.append(cluster_num[prev_num_samples:prev_num_samples+spike_counts[rec]])

            #print(np.shape(time_samples_each_rec))
            #print(time_samples_each_rec)
            
        with h5py.File(song_file, 'r') as f:
            motif_starts = f['event_types/singing/motiff_1/time_samples'].value  #start time of each motif
            motif_recs = f['event_types/singing/motiff_1/recording'].value  #recording number of each motif
            songtuples = zip(motif_recs, motif_starts)
            
            tot_motifs += len(songtuples)

            for rec, ideal_start in songtuples:
                
                neural_ideal_start = ideal_start+bin_size*(start_extend_bins+1-num_lookbacks)
                neural_ideal_end = neural_ideal_start+neural_song_length
                
                song_ideal_start = ideal_start+bin_size*start_extend_bins
                song_ideal_end = song_ideal_start+extended_song_length
                
                #process wav file into mel spec
                wav_name = 'experiment-rec_0%02d.mic.wav' % (rec)
                wav_file = os.path.join(directory, wav_name)
                
                wav_rate, wav_data = wavfile.read(wav_file)
                wav_data = wav_data[song_ideal_start:song_ideal_end]
                wav_data = butter_bandpass_filter(wav_data, lowcut, highcut, wav_rate, order=1)
                
                if len(wav_data)!=extended_song_length:
                    raise ValueError('less wav datapointes loaded than expected')
                
                #plot each filtered unmel-edwav
                fig = plt.figure()
                plt.plot(wav_data)
                plt.title(str(rec)+': '+str(song_ideal_start))
                plt.savefig(os.path.join(recording_folder_save, 'motif '+str(rec)+' '+str(ideal_start)+'.png'))
                plt.close(fig)
                
                #mel and compile into a list based on time bins
                wav_spectrogram = pretty_spectrogram(wav_data.astype('float64'), fft_size = fft_size, 
                                   step_size = step_size, log = True, thresh = spec_thresh)
                mel_list = list(make_mel(wav_spectrogram, mel_filter, shorten_factor = shorten_factor).transpose())
                
                current_t_list = time_samples_each_rec[rec] #extract neural activities for current motif
                current_cluster_list = cluster_each_rec[rec]  #extract cluster info for current motif

                neural_start_index = bisect.bisect_left(current_t_list, neural_ideal_start) #locate the index of starting point of the current motif
                neural_start_val = current_t_list[neural_start_index] #the actual starting point of the current motif
                #start_empty_bins = (neural_start_val-neural_ideal_start)//bin_size
                
                neural_end_index = bisect.bisect_left(current_t_list, neural_ideal_end)-1 #the end index of the current motif wrt indexing within the entire recording
                #end_empty_bins = (neural_ideal_end-current_t_list[neural_end_index])//bin_size

                t_during_song = current_t_list[neural_start_index:neural_end_index+1]-neural_ideal_start#normalize time sequence
                cluster_during_song = current_cluster_list[neural_start_index:neural_end_index+1] #extract cluster sequence

                counts_list = list() #contains all counts data for this motif
                for i in range(num_bins+num_lookbacks):
                    bin_start_index = bisect.bisect_left(t_during_song, i*bin_size)
                    bin_end_index = bisect.bisect_left(t_during_song,(i+1)*bin_size)
                    counts, bins = np.histogram(cluster_during_song[bin_start_index:bin_end_index], bins=np.arange(0,num_clusters+1))
                    counts_list.append(counts)
                    
                for i in range(num_bins):
                    mel_neural_tuple = (counts_list[i:i+10], mel_list[i])
                    mel_neural_comb.append(mel_neural_tuple)

                """for cluster in range(num_clusters):
                    index_this_cluster = [i for i, j in enumerate(cluster_during_song) if j == cluster]
                    t_this_cluster = t_during_song[index_this_cluster]
                    counts,bins = np.histogram(t_this_cluster, bins=[start_val:bin_size:end_val, sys.maxint])
                    """

                if counts_list[0].shape[0] != num_clusters:
                    raise ValueError('check cluster histogram')
                    
                if len(counts_list)-num_lookbacks != len(mel_list):
                    raise ValueError('Neural spiking bins do not correspond to spectrogram bins.')

                    #check for counting errors
        print('Finished.')
        
    if len(mel_neural_comb)!= num_bins*tot_motifs:
        raise ValueError('Might be missing some motifs or bins')
        
    pickle.dump(mel_neural_comb, open(os.path.join(bird_folder_save, save_name),'wb'))
        
    return mel_neural_comb


# In[6]:

def divide_parts(num_bins, divisor):
    num_bins_each_part_min = num_bins//divisor
    num_bins_each_part_max = num_bins_each_part_min + (num_bins%divisor!=0)
    
    num_bins_each_part = list([num_bins_each_part_max])*int(num_bins%divisor)+list([num_bins_each_part_min])*int(divisor-num_bins%divisor)
    
    return num_bins_each_part

def find_start_end_index(num_list):
    start_index = list()
    end_index = list()
    for i in range(len(num_list)):
        start_index.append(sum(num_list[:i]))
        end_index.append(sum(num_list[:i+1]))
    return start_index, end_index


# In[7]:

def plotspec(spec, vmin=-4, vmax=0):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
    cax = ax.matshow(spec, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower',vmax=vmax, vmin=vmin)
    fig.colorbar(cax)
    #plt.title('Mel Spectrogram')


# In[8]:

def spec2wav(example_spec, mel_inversion_filter, spec_thresh, shorten_factor, fft_size, step_size):
    example_inverted_spectrogram = mel_to_spectrogram(example_spec, mel_inversion_filter,
                                                    spec_thresh=5,
                                                    shorten_factor=1)
    inverted_mel_audio = invert_pretty_spectrogram(np.transpose(example_inverted_spectrogram), fft_size = 2048,
                                                step_size = 128, log = True, n_iter = 10)
    #plt.plot(inverted_mel_audio)
    return inverted_mel_audio


# In[9]:

def sort_songs(test_output_compiled, test_spec_compiled, num_songs, num_bins, divisor, break_song=False, test_indeces=[0]):
    predicted_song_compiled = list()
    original_song_compiled = list()
    rmse = list()

    if break_song:
        test_indeces = assign_test_indeces(divisor, break_song=break_song)
        num_bins_per_part = divide_parts(num_bins, divisor)
        for song_index in range(num_songs):
            predicted_song = list()
            original_song = list()
            for test_index in test_indeces:
                num_bins_this_index = num_bins_per_part[test_index]
                current_predicted_seg = test_output_compiled[test_index][song_index*num_bins_this_index:(song_index+1)*num_bins_this_index]
                current_original_seg = test_spec_compiled[test_index][song_index*num_bins_this_index:(song_index+1)*num_bins_this_index]
                predicted_song += list(current_predicted_seg)
                original_song += list(current_original_seg)
            rmse.append(np.sqrt(np.mean(np.square(np.array(predicted_song)-np.array(original_song)))))
            predicted_song_compiled.append(predicted_song)
            original_song_compiled.append(original_song)
    else:
        num_songs_each_part = divide_parts(num_songs, divisor)
        if len(test_indeces)>1:
            start_index, end_index = find_start_end_index(num_songs_each_part)
            for test_index in test_indeces:
                for song_index in range(num_songs_each_part[test_index]):
                    #offset = start_index[test_index]*num_bins
                    
                    predicted_song = test_output_compiled[test_index][song_index*num_bins:(song_index+1)*num_bins]
                    predicted_song_compiled.append(predicted_song)
                    original_song = test_spec_compiled[test_index][song_index*num_bins:(song_index+1)*num_bins]
                    original_song_compiled.append(original_song)
                    rmse.append(np.sqrt(np.mean(np.square(np.array(predicted_song)-np.array(original_song)))))
        else:
            test_index=test_indeces
            for song_index in range(num_songs_each_part[test_index[0]]):
                predicted_song = test_output_compiled[0][song_index*num_bins:(song_index+1)*num_bins]
                predicted_song_compiled.append(predicted_song)
                original_song = test_spec_compiled[0][song_index*num_bins:(song_index+1)*num_bins]
                original_song_compiled.append(original_song)
                rmse.append(np.sqrt(np.mean(np.square(np.array(predicted_song)-np.array(original_song)))))
    print([max(rmse),min(rmse)])
    
    return predicted_song_compiled, original_song_compiled, rmse

def assign_test_indeces(divisor, break_song=False):
    if break_song:
        test_indeces = range(divisor)
    else:
        test_indeces = [random.randint(0, divisor-1)]
    return test_indeces


# In[10]:

def eval_performance(x, y=None, mode = 'control_w', fft_size = 2048, step_size = 128, log = True, thresh = 5, shorten_factor = 1, 
              n_mel_freq_components = 64, start_freq = 200, end_freq = 15000, matric = 'corr'):
    
    if not y and not mode:
        raise ValueError('Either give a y or specify a mode.')
    #if mode and mode is not 'control' and mode is not 'self':
    #    raise ValueError('Mode should be control or self.')
    if y and mode:
        mode = None
        print('Mode has been ignored due to the presence of y values.')
    
    
    if matric is not 'corr' and matric is not 'rmse' and matric is not 'both':
        raise ValueError('mode should be set to either corr or rmse or both.')
        
    coeff = list()
    rmse = list()
    mel_filter, mel_inversion_filter = create_mel_filter(fft_size = fft_size,
                                                        n_freq_components = n_mel_freq_components,
                                                        start_freq = start_freq,
                                                        end_freq = end_freq)
    if mode == 'self':
        xnew = list()
        ynew = list()
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                xnew.append(x[i])
                ynew.append(x[j])
        x = xnew
        y = ynew
        
    for i in range(len(x)):
        #plt.plot(original)
        xi = np.array(x[i]).flatten()
        if not y:
            if mode == 'control_w':
                original_audio = spec2wav(np.array(x[i]).transpose(), mel_inversion_filter, thresh, shorten_factor, fft_size, step_size)
                control_audio = np.random.normal(0, np.std(original_audio), len(original_audio))
                #plt.plot(control_audio)
                yi = pretty_spectrogram(control_audio.astype('float64'), fft_size = fft_size, step_size = step_size, log = log, 
                                                   thresh = thresh)
                yi = make_mel(yi, mel_filter, shorten_factor = shorten_factor).flatten()
                
            if mode == 'control_f':
                original_audio = spec2wav(np.array(x[i]).transpose(), mel_inversion_filter, thresh, shorten_factor, fft_size, step_size)
                control_audio = np.array([random.choice([1, -1])*wavbit for wavbit in original_audio])
                yi = pretty_spectrogram(control_audio.astype('float64'), fft_size = fft_size, step_size = step_size, log = log, 
                                                   thresh = thresh)
                yi = make_mel(yi, mel_filter, shorten_factor = shorten_factor).flatten()
                
        else:
            yi = np.array(y[i]).flatten()
        if matric == 'corr':
            coeff.append(np.corrcoef([xi, yi])[0][1])
        if matric == 'rmse':
            rmse.append(np.sqrt(np.mean(np.square(np.array(xi)-np.array(yi)))))
        if matric == 'both':
            coeff.append(np.corrcoef([xi, yi])[0][1])
            rmse.append(np.sqrt(np.mean(np.square(np.array(xi)-np.array(yi)))))
            
    if matric == 'corr':
        return([round(np.mean(coeff, axis=0),4), round(np.std(coeff, axis=0),4)])
    if matric == 'rmse':
        return([round(np.mean(rmse, axis=0),4), round(np.std(rmse, axis=0),4)])
    if matric =='both':
        return([round(np.mean(rmse, axis=0),4), round(np.std(rmse, axis=0),4),
               round(np.mean(coeff, axis=0),4), round(np.std(coeff, axis=0),4)])


# In[11]:

def save_wav(waveform, rate, name, amp_rate=30000):
    waveform_adjusted = np.asarray(waveform/max(waveform)*amp_rate, dtype=np.int16)
    scipy.io.wavfile.write(name, rate, waveform_adjusted)


# In[ ]:



