# extract tensorboard data (audio only)
#
# Zhenhao Ge, 2024-06-07

import os
from pathlib import Path
import glob
import numpy as np
import datetime
import argparse
# import tensorflow as tf
# from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import wave

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

def parse_sample(sample):

    # get data time string
    dt = datetime.datetime.fromtimestamp(sample[0])
    dt_str = dt.strftime('%Y%m%d-%H%M%S')

    # get step
    step = sample[1]

    # get data
    data = np.fromstring(sample[2], 'int16')
    
    # get ext
    ext = sample[3].split('/')[-1]

    return data, step, dt_str, ext

def audiowrite(audiofile, data, params):
    """ write audio file """

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

def parse_args():
    usage = 'extract data from tensorboard (currently audio files only)'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--log-path', type=str, help='tensorboard log dir')
    parser.add_argument('--output-path', type=str, help='data output dir')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # work_path = os.getcwd()
    # dataset = 'LJSpeech'
    # log_folder = '20240603.stage-2.titan13' # '20240530.stage-1.titan13', '20240603.stage-2.titan13'
    # args.log_path = os.path.join(work_path, 'Models', dataset, 'tensorboard', log_folder)
    # args.output_path = os.path.join(work_path, 'Outputs', 'Training', dataset, log_folder)

    # localize input arguments
    log_path = args.log_path
    output_path = args.output_path

    # check dir existance
    assert os.path.isdir(log_path), 'log dir: {} does not exist!'.format(log_path)

    # create output dir (if needed)
    if os.path.isdir(output_path):
        print('use existing output dir: {}'.format(output_path))
    else:
        os.makedirs(output_path)
        print('created new output dir: {}'.format(output_path))  

    # get event file paths under the selected log dir
    event_filepaths = sorted(glob.glob(os.path.join(log_path, "events.out.tfevents.*")))

    for n, f in enumerate(event_filepaths):

        f = event_filepaths[n]

        print('event path: {}'.format(f))

        # load event file
        event_acc = EventAccumulator(f)
        event_acc.Reload()

        # get tags
        tags = event_acc.Tags()['audio']
        ntags = len(tags)

        # if n > 3 and ntags > 0:
        #     break

        for i in range(ntags):

            tag = tags[i]
            print('tag: {}'.format(tag))

            # get raw data for the current tag
            raw = event_acc.Audio(tag)
            ndata = len(raw)

            for j in range(ndata):

                # get current sample
                sample = raw[j]

                # parse sample
                data, step, dt_str, ext = parse_sample(sample)

                # construct parameters
                nframes = len(data)
                params = [1, 2, 24000, nframes, 'NONE', 'not compressed']
                
                # set output audio file path
                out_filename = '{}_{:03d}.{}'.format(tag.replace('/', '_'), step, ext)
                out_filepath = os.path.join(output_path, out_filename)

                # write output audio file
                audiowrite(out_filepath, data, params)
                print('wrote {}'.format(out_filepath))
