#!/usr/bin/env python3
#
# FFProbe Bitrate Graph
#
# Copyright (c) 2013-2019, Eric Work
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

import sys
import shutil
import argparse
import subprocess
import multiprocessing
import math
import collections
import statistics
from enum import Enum

# prefer C-based ElementTree
try:
    import xml.etree.cElementTree as etree
except ImportError:
    import xml.etree.ElementTree as etree

# check for matplot lib
try:
    import numpy
    import matplotlib.pyplot as matplot
except ImportError:
    sys.stderr.write("Error: Missing package 'python3-matplotlib'\n")
    sys.exit(1)

# check for ffprobe in path
if not shutil.which("ffprobe"):
    sys.stderr.write("Error: Missing ffprobe from package 'ffmpeg'\n")
    sys.exit(1)

# get list of supported matplotlib formats
format_list = list(
    matplot.figure().canvas.get_supported_filetypes().keys())
matplot.close()  # destroy test figure

# parse command line arguments
parser = argparse.ArgumentParser(
    description="Graph bitrate for audio/video stream")
parser.add_argument('input', help="input file/stream", metavar="INPUT")
parser.add_argument('-s', '--stream', help="stream type (default: video)",
                    choices=["audio", "video"], default="video")
parser.add_argument('-o', '--output', help="output file")
parser.add_argument('-f', '--format', help="output file format",
                    choices=format_list)
parser.add_argument('-p', '--progress', help="show progress",
                    action='store_true')
parser.add_argument('--min', help="set plot minimum (kbps)", type=int)
parser.add_argument('--max', help="set plot maximum (kbps)", type=int)
parser.add_argument('-t', '--show-frame-types',
                    help="shot bitrate of different frame types",
                    action='store_true')
parser.add_argument(
    '--max-display-values', 
    help="set the maximum number of values shown on the x axis. " + 
    "will downscale if video length is higher than given value. " + 
    "for no downscaling set to 0. not compatible with option --show-frame-types " + 
    "(default: 700)", 
    type=int,
    default=700)
args = parser.parse_args()

# check if format given w/o output file
if args.format and not args.output:
    sys.stderr.write("Error: Output format requires output file\n")
    sys.exit(1)

# check given y-axis limits
if args.min and args.max and (args.min >= args.max):
    sys.stderr.write("Error: Maximum should be greater than minimum\n")
    sys.exit(1)

# set ffprobe stream specifier
if args.stream == 'audio':
    stream_spec = 'a'
elif args.stream == 'video':
    stream_spec = 'V'
else:
    raise RuntimeError("Invalid stream type")

Frame = collections.namedtuple('Frame', ['time', 'size_kbit', 'type'])

class Color(Enum):
    I = 'red'
    P = 'green'
    B = 'blue'
    AUDIO = 'indianred'
    FRAME = 'indianred'

def open_ffprobe_get_format(filepath):
    return subprocess.Popen(
        ["ffprobe",
            "-show_entries", "format",
            "-print_format", "xml",
            filepath
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL)

def open_ffprobe_get_frames(filepath, stream_selector):
    return subprocess.Popen(
        ["ffprobe",
            "-select_streams", stream_selector,
            "-threads", str(multiprocessing.cpu_count()),
            "-print_format", "xml",
            "-show_entries", 
            "frame=pict_type,pkt_duration_time,pkt_pts_time," + 
            "best_effort_timestamp_time,pkt_size",
            filepath
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL)

def read_total_time_from_format_xml(source):
    format_data = etree.parse(source)
    format_elem = format_data.find('.//format')
    return float(format_elem.get('duration'))

def read_frame_data(source_iterable, frame_read_callback):
    data = []
    frame_count = 0
    for event in source_iterable:

        # skip non-frame elements
        node = event[1]
        if node.tag != 'frame':
            continue

        frame_count += 1
        frame_type = node.get('pict_type')

        # collect frame data
        try:
            frame_time = float(node.get('best_effort_timestamp_time'))
        except:
            try:
                frame_time = float(node.get('pkt_pts_time'))
            except:
                if frame_count > 1:
                    frame_time += float(node.get('pkt_duration_time'))

        frame_size_in_kbit = (float(node.get('pkt_size')) * 8 / 1000)

        frame = Frame(frame_time, frame_size_in_kbit, frame_type)
        data.append(frame)

        if frame_read_callback is not None:
            frame_read_callback(frame)
    
    return data

def group_frames_to_seconds(frames, seconds_start, seconds_end, seconds_step=1):
    # create an index for lookup performance
    frames_indexed_by_seconds = {}
    for frame in frames:
        second = math.floor(frame.time)
        if second not in frames_indexed_by_seconds:
            frames_indexed_by_seconds[second] = []
        frames_indexed_by_seconds[second].append(frame)

    # iterate over seconds with the given step
    mapped_data = {}
    for second in range(seconds_start, seconds_end, seconds_step):
        # if steps is greater than one, this will run over each second in between
        second_substep = second
        while second + seconds_step > second_substep:
            # take the highest bitrate of all seconds in between
            size_sum = sum(
                frame.size_kbit for frame in
                frames_indexed_by_seconds.get(second_substep, ()))
            
            mapped_data[second] = max(
                mapped_data.get(second, 0), size_sum)
            
            second_substep += 1
    return mapped_data


total_time = None
from_xml = args.input.endswith('.xml')

# read total time from format
try:
    if from_xml:
        total_time = read_total_time_from_format_xml(args.input)
    else:
        proc = open_ffprobe_get_format(args.input)
        total_time = read_total_time_from_format_xml(proc.stdout)
except:
    sys.stderr.write("Error: Failed to determine stream duration\n")
    sys.exit(1)

# read frames from file
if from_xml:
    frames_source = etree.iterparse(args.input)
else:
    proc_frame = open_ffprobe_get_frames(args.input, stream_spec)
    frames_source = etree.iterparse(proc_frame.stdout)

if args.progress:
    last_percent = 0
    def report_progress(frame):
        global last_percent
        percent = math.floor((frame.time / total_time) * 100.0)
        if percent > last_percent:
            sys.stdout.write("\rProgress: {:2}%".format(percent))
            last_percent = percent
    report_func = report_progress
else:
    report_func = None

frames_raw = read_frame_data(frames_source, report_func)
if args.progress:
    print(flush=True)

# check for success
if len(frames_raw) == 0:
    sys.stderr.write("Error: No frame data, failed to execute ffprobe\n")
    sys.exit(1)

# setup new figure
matplot.figure(figsize=[10, 4]).canvas.set_window_title(args.input)
matplot.title("Stream Bitrate over Time")
matplot.xlabel("Time (sec)")
matplot.ylabel("Bitrate (kbit/s)")
matplot.grid(True, axis='y')
matplot.tight_layout()

total_time_last_second = math.floor(total_time)
global_peak_bitrate = 0
global_mean_bitrate = 0
bars = {}

# add a statcking bar for each frame type
if args.show_frame_types and args.stream == 'video':

    sums_of_values = ()

    for frame_type in ['I', 'B', 'P']:
        filtered_frames = [frame for frame in frames_raw 
                           if frame.type == frame_type]

        if len(filtered_frames) == 0:
            continue

        seconds_with_bitrates = group_frames_to_seconds(
            filtered_frames, 0, total_time_last_second)

        bar = matplot.bar(
                seconds_with_bitrates.keys(), 
                seconds_with_bitrates.values(),
                bottom=sums_of_values if len(sums_of_values) > 0 else 0,
                color=Color[frame_type].value,
                width=1)
        
        bars[frame_type] = bar
        
        # add current bitrate values to all previous
        # needed so that the stacking bars know their min value
        if len(sums_of_values) == 0:
            sums_of_values = list(seconds_with_bitrates.values())
        else:
            sums_of_values = [x + y for x, y in zip(
                sums_of_values, seconds_with_bitrates.values())]

    global_peak_bitrate = max(sums_of_values)
    global_mean_bitrate = statistics.mean(sums_of_values)

else:
    second_steps = max(1, math.floor(total_time / args.max_display_values)) \
                   if args.max_display_values >= 1 else 1
    seconds_with_bitrates = group_frames_to_seconds(
        frames_raw, 0, total_time_last_second, second_steps)

    if args.stream == 'audio':
        color = Color.AUDIO.value
    elif args.stream == 'video':
        color = Color.FRAME.value
    else:
        color = 'black'

    matplot.bar(
        seconds_with_bitrates.keys(), 
        seconds_with_bitrates.values(),
        color=color,
        width=second_steps)

    seconds_with_bitrates = group_frames_to_seconds(
        frames_raw, 0, total_time_last_second)
    global_peak_bitrate = max(seconds_with_bitrates.values())
    global_mean_bitrate = statistics.mean(seconds_with_bitrates.values())

# set y-axis limits if requested
if args.min:
    matplot.ylim(ymin=args.min)
if args.max:
    matplot.ylim(ymax=args.max)

# calculate peak line position (left 15%, above line)
peak_text_x = matplot.xlim()[1] * 0.15
peak_text_y = global_peak_bitrate + \
    ((matplot.ylim()[1] - matplot.ylim()[0]) * 0.015)
peak_text = "peak ({:.0f})".format(global_peak_bitrate)

# draw peak as think black line w/ text
matplot.axhline(global_peak_bitrate, linewidth=1.5, color='black')
matplot.text(peak_text_x, peak_text_y, peak_text,
             horizontalalignment='center', fontweight='bold', color='black')

# calculate mean line position (right 85%, above line)
mean_text_x = matplot.xlim()[1] * 0.85
mean_text_y = global_mean_bitrate + \
    ((matplot.ylim()[1] - matplot.ylim()[0]) * 0.015)
mean_text = "mean ({:.0f})".format(global_mean_bitrate)

# draw mean as think black line w/ text
matplot.axhline(global_mean_bitrate, linewidth=1.5, color='black')
matplot.text(mean_text_x, mean_text_y, mean_text,
             horizontalalignment='center', fontweight='bold', color='black')

if len(bars) > 1:
    matplot.legend(bars.values(), bars.keys())

# render graph to file (if requested) or screen
if args.output:
    matplot.savefig(args.output, format=args.format, dpi=300)
else:
    matplot.show()