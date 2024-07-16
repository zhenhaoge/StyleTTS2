import os
import numpy as np
import shutil
from pydub import AudioSegment
import contextlib
import pyaudio
import wave
import pyrubberband
import librosa
import soundfile as sf
from audiostretchy.stretch import stretch_audio

def audioread(audiofile, starttime=0.0, duration=float('inf'), verbose=False):
  """
  read audio with specified starting time and duration
  default settings:
      startfime: 0, start 0.0 sec (at the beginning)
      duration: float(''inf), duration in time (sec.) covers the entire audio file
  """

  f = wave.open(audiofile)

  # get parameters (nchannels, sampwidth, framerate, nframes, comptype, compname)
  params = list(f.getparams())
  framerate = params[2] # sampling rate, e.g. 16000, or 8000
  nframes = params[3]

  # skip frames before starting frame
  startframe = round(framerate*starttime)
  f.setpos(startframe)

  # get the number of frames/samples actually to be read
  if duration < float('inf'):
    nframes_to_read= int(min(round(framerate*duration), nframes-startframe))
  else:
    nframes_to_read = int(nframes - startframe)

  # update the # of frames to be the # of frames to be read only
  params[3] = nframes_to_read

  # read frames
  data = f.readframes(nframes_to_read)
  data = np.fromstring(data, 'int16')

  # close file
  f.close()

  if verbose:
      time_to_read = nframes_to_read/framerate
      endtime = starttime + time_to_read
      endframe = startframe + nframes_to_read
      print('read ' + '%.2f' % time_to_read + ' sec.: ' + '%.2f' % starttime +
            ' ~ ' + '%.2f' % endtime + ' sec. (' + str(startframe) + ' ~ ' +
            str(endframe) + ' frame)')

  return data, params

def audiowrite(audiofile, data, params):
  """
  write audio file
  """

  # make sure the nframes matches the data length
  # so no need to update 'params' before calling this function
  params[3] = len(data)

  # enable to read scaled data
  if not isinstance(data[0], np.int16):
    dmax = 2 ** (params[1]*8-1)
    data = np.asarray([int(i) for i in data*dmax], dtype='int16')

  # write data
  f = wave.open(audiofile, 'w')
  f.setparams(tuple(params))
  f.writeframes(data)
  f.close()

def audioplay(audiofile, chunktime=0.05, starttime=0.0, duration=float('inf'),
              showprogress=True, verbose=False):
    """
    play audio with specified starting time and duration
    default settings:
        chunktime: 0.05, load audio 0.05 sec at a time
        startfime: 0, start 0.0 sec (at the beginning)
        duration: float(''inf), duration in time (sec.) covers the entire audio file
    """

    f = wave.open(audiofile, 'r')
    p = pyaudio.PyAudio()

    # get parameters
    sampwidth = f.getsampwidth() # sample width in bytes, e.g. 2
    nchannels = f.getnchannels() # 1 for mono, 2 for stereo
    framerate = f.getframerate() # sampling rate, e.g. 16000, or 8000
    nframes = f.getnframes()

    # open stream
    stream = p.open(format = p.get_format_from_width(sampwidth),
                    channels = nchannels, rate = framerate,
                    output = True)

    chunksize = round(framerate*chunktime)

    # skip frames before starting frame
    startframe = round(framerate*starttime)
    f.setpos(startframe)

    # get the number of frames/samples actually to be played
    if duration < float('inf'):
        nframes_to_play= min(round(framerate*duration), nframes-startframe)
    else:
        nframes_to_play = nframes - startframe

    # read and play audio data
    nchunks = int(nframes_to_play/chunksize)
    lastchunk = nframes - nchunks*chunksize
    if lastchunk > 0:
        additional_chunk = 1

    # initiate the progress bar
    if showprogress:
        bar = progressbar.ProgressBar(maxval=nchunks + additional_chunk, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

    # loop over chunks except the last one
    for i in range(nchunks):
        #print('chunk ' + str(i) + ': ' + str(i*chunksize) + ' ~ ' + str((i+1)*chunksize-1))
        data = f.readframes(chunksize)
        stream.write(data)
        if showprogress: bar.update(i+1)

    # the last one chunk
    if lastchunk > 0:
        #print('chunk ' + str(i+1) + ': ' + str((i+1)*chunksize) + ' ~ ' + str(nframes_to_play-1))
        data = f.readframes(chunksize)
        stream.write(data)
        if showprogress: bar.update(i+1)

    # close bar
    if showprogress: bar.finish()

    #stop stream
    stream.stop_stream()
    stream.close()

    # close PyAudio
    p.terminate()

    # close file
    f.close()

    if verbose:
        #print('audio duration: ' + '%.2f' % (nframes/framerate) + ' sec. (' + str(nframes) + ' frames)')
        time_to_play = nframes_to_play/framerate
        endtime = starttime + time_to_play
        endframe = startframe + nframes_to_play
        #print('\n')
        print('played ' + '%.2f' % time_to_play + ' sec.: ' + '%.2f' % starttime +
              ' ~ ' + '%.2f' % endtime + ' sec. (' + str(startframe) + ' ~ '
              + str(endframe) + ' frame)')

def soundsc(data, para, dmax=0, nchunks=20, showprogress=True):
  """
  play scaled sound given data frame
  """

  p = pyaudio.PyAudio()

  # get parameters
  nchannels = para[0]
  sampwidth = para[1]
  framerate = para[2]
  nframes = len(data)

  # open stream
  stream = p.open(format = p.get_format_from_width(sampwidth),
           channels = nchannels, rate = framerate,
           output = True)

  # set the default dmax
  if dmax == 0:
    dmax = 2 ** (sampwidth*8-1)

  # scale data
  data_raw = np.asarray([int(i) for i in data*dmax], dtype='int16')

  # cut into chunks
  nsecs = nframes/framerate
  if nsecs > nchunks:
    chunksize = int(nframes/nchunks)
  else:
    nchunks = int(np.ceil(nsecs))
    chunksize = framerate
  nframes_in_chunk = [chunksize] * nchunks
  nframes_in_chunk[-1] = nframes - (nchunks-1)*chunksize

  # initiate the progress bar
  if showprogress:
    bar = progressbar.ProgressBar(maxval=nchunks, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

  # play the sound
  for i in range(nchunks):
    if i != nchunks-1:
      #print('chunk ' + str(i+1) + '/' + str(nchunks) + ': [' +
      #      str(i*chunksize) + ' ~ ' + str((i+1)*chunksize) + ') ...')
      stream.write(data_raw[i*chunksize:(i+1)*chunksize], nframes_in_chunk[i])
    else:
      #print('chunk ' + str(i+1) + '/' + str(nchunks) + ': [' +
      #      str(i*chunksize) + ' ~ ' + str(nframes) + ') ...')
      stream.write(data_raw[i*chunksize:], nframes_in_chunk[i])
    if showprogress: bar.update(i+1)

  # close bar
  if showprogress: bar.finish()

def wav_duration(filename):
  """
  get wav file duration in seconds
  """
  with contextlib.closing(wave.open(filename,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    #print(duration)
  return duration

def change_speed_only(sound, tempo_ratio):
    y = np.array(sound.get_array_of_samples())
    if sound.channels == 2:
        y = y.reshape((-1, 2))

    sample_rate = sound.frame_rate
    y_fast = pyrubberband.time_stretch(y, sample_rate, tempo_ratio)

    channels = 2 if (y_fast.ndim == 2 and y_fast.shape[1] == 2) else 1
    y = np.int16(y_fast * 2 ** 15)

    new_seg = AudioSegment(y.tobytes(), frame_rate=sample_rate, sample_width=2, channels=channels)

    return new_seg

def list_flat(l):
  """convert 2-layer list to 1-layer (flatern list)"""
  return  [item for sublist in l for item in sublist]

def extract_wav_channel(wav_in, wav_out, channel=0, verbose=False):

  if channel=='left' or channel=='l': channel=0
  if channel=='right' or channel=='r': channel=1

  with wave.open(wav_in, 'r') as f:
    params = list(f.getparams())
    nchannels = params[0]
    sampwidth = params[1]
    nframes = params[3]
    data = f.readframes(nframes) # range: [0,2^(4*sampwidth)-1]

  if channel+1 > nchannels:
    raise Exception('No channel {} since {} has {} channels!'.format(channel, \
                    wav_in, nchannels))

  samples = [[] for i in range(sampwidth)]
  samples_in_channel = [[] for i in range(nchannels)]
  for i in range(sampwidth):
    samples[i] = data[i::sampwidth] # samples[0] <-- data[0::sampwidth]
    samples_in_channel[i] = samples[i][channel::nchannels]

  data_in_channel = bytes(list_flat(list(map(list, zip(*samples_in_channel)))))

  with wave.open(wav_out, 'w') as f:
    params[0] = 1
    params[3] = len(data_in_channel)
    f.setparams(tuple(params))
    f.writeframes(data_in_channel)
    if verbose:
      print('wrote {} (channel {}) to {}'.format(wav_in, channel, wav_out))

def adjust_speed(input_wavfile, output_wavfile, speed, verbose=False):

    # get ratio from speed
    # new audio duration / original audio duration (<1 means faster)
    ratio = 1 / speed

    output_wavfile_temp = output_wavfile.replace('.wav', '.tmp.wav')
    stretch_audio(input_wavfile, output_wavfile_temp, ratio)

    # get the input and output (temp) wavs
    input_wav, sr = librosa.load(input_wavfile, sr=None)
    output_wav_temp, sr2 = librosa.load(output_wavfile_temp, sr=None)
    assert sr2 == sr, 'input and output wav file have different sampling rate!'
    del sr2

    # get the input and output (temp) #samples
    input_nsamples = len(input_wav)
    output_nsamples_temp = len(output_wav_temp)

    # get the output #samples (should be based on the time-scaling factor)
    output_nsamples = int(np.ceil(input_nsamples/speed))

    # truncate the trailing silence if needed
    output_dur_temp = output_nsamples_temp / sr
    output_dur = output_nsamples / sr
    if output_nsamples_temp > output_nsamples:
        if verbose:
            print('truncating the trailing silence: {:.3f} sec. -> {:.3f} sec.'.format(
                output_dur_temp, output_dur))
        output_wav = output_wav_temp[:output_nsamples]
        sf.write(output_wavfile, output_wav, sr)
        os.remove(output_wavfile_temp)
    else:
        if verbose:
            print('output tmp dur: {:.3f}, output dur: {:.3f}, no truncation needed'.format(
                output_dur_temp, output_dur))
        shutil.move(output_wavfile_temp, output_wavfile)

    return input_wav, output_wav, sr    

def convert_opus2wav(opus_file, wav_file, target_sr=16000, rm_opus=False):
    cmd = f'ffmpeg -y -i {opus_file} -ac 1 -ar {target_sr} {wav_file}'
    try:
      os.system(cmd)
    except:
      sys.exit(f'Failed to run the cmd: {cmd}')
    if rm_opus is True:
      os.remove(opus_file)

