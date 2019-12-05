#!/usr/bin/env python3
#
# FFProbe Bitrate Graph
#
# Original work Copyright (c) 2013-2019, Eric Work
# Modified work Copyright (c) 2019, Steve Schmidt
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
import csv
import datetime
from enum import Enum
from typing import Callable, Union, List, IO, Iterable, Optional

# prefer C-based ElementTree
try:
    import xml.etree.cElementTree as etree
except ImportError:
    import xml.etree.ElementTree as etree

# check for matplot lib
try:
    import numpy
    import matplotlib.pyplot as matplot
    import matplotlib
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

format_list.append("xml_raw")
format_list.append("csv_raw")

# parse command line arguments
parser = argparse.ArgumentParser(
    description="Graph bitrate for audio/video stream")
parser.add_argument("input", help="input file/stream", metavar="INPUT")
parser.add_argument("-s", "--stream", help="stream type (default: video)",
                    choices=["audio", "video"], default="video")
parser.add_argument("-o", "--output", help="output file")
parser.add_argument("-f", "--format", help="output file format",
                    choices=format_list)
parser.add_argument("-p", "--progress", help="show progress",
                    action="store_true")
parser.add_argument("--min", help="set plot minimum (kbps)", type=int)
parser.add_argument("--max", help="set plot maximum (kbps)", type=int)
parser.add_argument("-t", "--show-frame-types",
                    help="shot bitrate of different frame types",
                    action="store_true")
parser.add_argument(
    "--max-display-values", 
    help="set the maximum number of values shown on the x axis. " + 
    "will downscale if video length is longer than the given value. " + 
    "for no downscaling set to -1. not compatible with option --show-frame-types " + 
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
if args.stream == "audio":
    stream_spec = "a"
elif args.stream == "video":
    stream_spec = "V"
else:
    raise RuntimeError("Invalid stream type")

# datatype for raw frame data
Frame = collections.namedtuple("Frame", ["time", "size_kbit", "type"])

class Color(Enum):
    I = "red"
    P = "green"
    B = "blue"
    AUDIO = "indianred"
    FRAME = "indianred"


def open_ffprobe_get_format(filepath: str) -> subprocess.Popen:
    """ Opens an ffprobe process that reads the format data
    for filepath and returns the process.
    """
    return subprocess.Popen(
        ["ffprobe",
            "-hide_banner",
            "-loglevel", "error",
            "-show_entries", "format",
            "-print_format", "xml",
            filepath
        ],
        stdout=subprocess.PIPE)


def open_ffprobe_get_frames(
    filepath: str, stream_selector: str) -> subprocess.Popen:
    """ Opens an ffprobe process that reads all frame data for
    filepath and returns the process.
    """
    return subprocess.Popen(
        ["ffprobe",
            "-hide_banner",
            "-loglevel", "error",
            "-select_streams", stream_selector,
            "-threads", str(multiprocessing.cpu_count()),
            "-print_format", "xml",
            "-show_entries", 
            "frame=pict_type,pkt_duration_time,pkt_pts_time," + 
            "best_effort_timestamp_time,pkt_size",
            filepath
        ],
        stdout=subprocess.PIPE)


def save_raw_xml(filepath: str, target_path: str, stream_selector: str) -> None:
    """ Reads all raw frame data from filepath and saves it to target_path. """
    if args.progress:
        last_percent = 0
        with open_ffprobe_get_format(args.input) as proc_format:
            total_media_time_in_seconds = read_total_time_from_format_xml(
                                            proc_format.stdout)

    with open(target_path, "wb") as f:
        # open and clear file
        f.seek(0)
        f.truncate()

        with subprocess.Popen(
            ["ffprobe",
                "-hide_banner",
                "-loglevel", "error",
                "-select_streams", stream_selector,
                "-threads", str(multiprocessing.cpu_count()),
                "-print_format", "xml",
                "-show_entries",
                "format:frame=pict_type,pkt_duration_time,pkt_pts_time," + 
                "best_effort_timestamp_time,pkt_size",
                filepath
            ], 
            stdout=subprocess.PIPE) as proc:
                # start process and iterate over output lines
                for line in iter(proc.stdout):
                    # if progress is enabled
                    # look for lines starting with frame tag
                    # try parsing the time from them and print percent
                    if args.progress and line.lstrip().startswith(b"<frame "):
                        try:
                            frame_time = try_get_frame_time_from_node(
                                etree.fromstring(line))
                            percent = math.floor((frame_time / 
                                total_media_time_in_seconds) * 100.0)
                            if percent > last_percent:
                                print_progress(percent)
                                last_percent = percent
                        except:
                            pass
                    f.write(line)
                if args.progress:
                    print(flush=True)


def save_raw_csv(raw_frames: List[Frame], target_path: str) -> None:
    """ Saves raw_frames as a csv file. """
    with open(target_path, "w") as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(Frame._fields)
        wr.writerows(raw_frames)


def read_total_time_from_format_xml(source: Union[str, IO]) -> float:
    """ Parses the source and returns the extracted total duration. """
    format_data = etree.parse(source)
    format_elem = format_data.find(".//format")
    return float(format_elem.get("duration"))


def try_get_frame_time_from_node(
        node: etree.ElementTree, frame_count: int = 0) -> Optional[float]:
    frame_time = None
    try:
        frame_time = float(node.get("best_effort_timestamp_time"))
    except:
        try:
            frame_time = float(node.get("pkt_pts_time"))
        except:
            if frame_count > 1:
                frame_time += float(node.get("pkt_duration_time"))

    return frame_time


def print_progress(percent: float) -> None:
    sys.stdout.write("\rProgress: {:2}%".format(percent))


def read_frame_data(
        source_iterable: Iterable, 
        frame_read_callback: Callable[[Frame], None]
        ) -> List[Frame]:
    """ Iterates over source_iterabe and creates a list of Frame objects.
    Will call frame_read_callback on the end of each loop so that progess
    can be calculated.
    """
    data = []
    frame_count = 0
    for event in source_iterable:

        # skip non-frame elements
        node = event[1]
        if node.tag != "frame":
            continue

        frame_count += 1

        frame_type = node.get("pict_type")
        frame_time = try_get_frame_time_from_node(node)
        frame_size_in_kbit = (float(node.get("pkt_size")) * 8 / 1000)
        frame = Frame(frame_time, frame_size_in_kbit, frame_type)
        data.append(frame)

        if frame_read_callback is not None:
            frame_read_callback(frame)
    
    return data


def group_frames_to_seconds(
        frames: List[Frame], 
        seconds_start: int, 
        seconds_end: int,
        seconds_step: int = 1):
    """ Iterates from seconds_start to seconds_end and groups
    all frame data by its whole second.
    The second is floored,
    so all data of the first second will be in second 0.
    If seconds_step is greater than 1, 
    the result will not contain every second in between.
    In this case, for each result second, the highest bitrate
    of this second and all up to the next will be chosen.

    Example:
    seconds_start=0, seconds_end=10, seconds_step=3

    bitrate of second 0 is 3400
    bitrate of second 1 is 5290
    bitrate of second 2 is 4999
    ...

    The result will contain second 0 with bitrate 5290 (the highest),
    but not second 1 or 2.
    The next result second will be 3, containing the highest bitrate
    of itself and the next 2 seconds, and so on.

    """
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


# if the output is raw xml, just call the function and exit
if args.format == "xml_raw":
    save_raw_xml(args.input, args.output, stream_spec)
    sys.exit(0)

total_media_time_in_seconds: float = None
source_is_xml: bool = args.input.endswith(".xml")

# read total time from format
try:
    if source_is_xml:
        total_media_time_in_seconds = read_total_time_from_format_xml(args.input)
    else:
        proc = open_ffprobe_get_format(args.input)
        total_media_time_in_seconds = read_total_time_from_format_xml(proc.stdout)
except:
    sys.stderr.write("Error: Failed to determine stream duration\n")
    sys.exit(1)

# open frame data reader for media or xml file
if source_is_xml:
    frames_source = etree.iterparse(args.input)
else:
    proc_frame = open_ffprobe_get_frames(args.input, stream_spec)
    frames_source = etree.iterparse(proc_frame.stdout)

if args.progress:
    # prepare a progress callback function
    last_percent = 0
    def report_progress(frame):
        global last_percent
        percent = math.floor((frame.time / total_media_time_in_seconds) * 100.0)
        if percent > last_percent:
            print_progress(percent)
            last_percent = percent
    report_func = report_progress
else:
    report_func = None

# read frame data
frames_raw: List[Frame] = read_frame_data(frames_source, report_func)

if args.progress:
    print(flush=True)

# check for success
if len(frames_raw) == 0:
    sys.stderr.write("Error: No frame data, failed to execute ffprobe\n")
    sys.exit(1)

# if the output is csv raw, write the file and we're done
if args.format == "csv_raw":
    save_raw_csv(frames_raw, args.output)
    sys.exit(0)

# prepare the chart and setup new figure
matplot.figure(figsize=[10, 4]).canvas.set_window_title(args.input)
matplot.title("Stream Bitrate over Time")
matplot.xlabel("Time")
matplot.ylabel("Bitrate (kbit/s)")
matplot.grid(True, axis="y")
matplot.tight_layout()

# set 10 x axes ticks
matplot.xticks(range(
    0, 
    int(total_media_time_in_seconds) + 1, 
    max(int(total_media_time_in_seconds / 10), 1)))

# format axes values
matplot.gcf().axes[0].xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, loc: datetime.timedelta(seconds=int(x))))  
matplot.gca().get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

total_media_time_in_seconds_floored: int = math.floor(total_media_time_in_seconds)
global_peak_bitrate: int = 0
global_mean_bitrate: int = 0
bars: dict = {}

if args.show_frame_types and args.stream == "video":
    # calculate bitrate for each frame type
    # and add a statcking bar for each

    sums_of_values = ()
    for frame_type in ["I", "B", "P", "?"]:
        filtered_frames = [frame for frame in frames_raw 
                           if frame.type == frame_type]
        if len(filtered_frames) == 0:
            continue

        # simple frame grouping to seconds
        seconds_with_bitrates = group_frames_to_seconds(
            filtered_frames, 0, total_media_time_in_seconds_floored)

        bar = matplot.bar(
                seconds_with_bitrates.keys(), 
                seconds_with_bitrates.values(),
                bottom=sums_of_values if len(sums_of_values) > 0 else 0,
                color=Color[frame_type].value if frame_type in dir(Color) else Color.FRAME.value,
                width=1)
        bars[frame_type] = bar
        
        # add current frame bitrate values to all previous
        # needed so that the stacking bars know their min value
        # and so that the peak and mean can be calculated
        if len(sums_of_values) == 0:
            sums_of_values = list(seconds_with_bitrates.values())
        else:
            sums_of_values = [x + y for x, y in zip(
                sums_of_values, seconds_with_bitrates.values())]

    global_peak_bitrate = max(sums_of_values)
    global_mean_bitrate = statistics.mean(sums_of_values)

else:
    # calculate how many seconds in between should be left out
    # so that only a maximum of 'args.max_display_values' number of values are shown
    second_steps = max(1, math.floor(total_media_time_in_seconds / args.max_display_values)) \
                   if args.max_display_values >= 1 else 1
    seconds_with_bitrates = group_frames_to_seconds(
        frames_raw, 0, total_media_time_in_seconds_floored, second_steps)

    if args.stream == "audio":
        color = Color.AUDIO.value
    elif args.stream == "video":
        color = Color.FRAME.value
    else:
        color = "black"

    matplot.bar(
        seconds_with_bitrates.keys(), 
        seconds_with_bitrates.values(),
        color=color,
        width=second_steps)

    # group the raw frames to seconds again
    # but this time without leaving out any seconds
    # to calculate the peak and mean
    seconds_with_bitrates = group_frames_to_seconds(
        frames_raw, 0, total_media_time_in_seconds_floored)
    global_peak_bitrate = max(seconds_with_bitrates.values())
    global_mean_bitrate = statistics.mean(seconds_with_bitrates.values())

# set y-axis limits if requested
if args.min:
    matplot.ylim(ymin=args.min)
if args.max:
    matplot.ylim(ymax=args.max)

# calculate peak line position (left 8%, above line)
peak_text_x = matplot.xlim()[1] * 0.08
peak_text_y = global_peak_bitrate + \
    ((matplot.ylim()[1] - matplot.ylim()[0]) * 0.015)
peak_text = "peak ({:,})".format(int(global_peak_bitrate))

# draw peak as think black line w/ text
matplot.axhline(global_peak_bitrate, linewidth=1.5, color="black")
matplot.text(peak_text_x, peak_text_y, peak_text,
             horizontalalignment="center", fontweight="bold", color="black")

# calculate mean line position (right 92%, above line)
mean_text_x = matplot.xlim()[1] * 0.92
mean_text_y = global_mean_bitrate + \
    ((matplot.ylim()[1] - matplot.ylim()[0]) * 0.015)
mean_text = "mean ({:,})".format(int(global_mean_bitrate))

# draw mean as think black line w/ text
matplot.axhline(global_mean_bitrate, linewidth=1.5, color="black")
matplot.text(mean_text_x, mean_text_y, mean_text,
             horizontalalignment="center", fontweight="bold", color="black")

if len(bars) >= 1:
    matplot.legend(bars.values(), bars.keys())

# render graph to file (if requested) or screen
if args.output:
    matplot.savefig(args.output, format=args.format, dpi=300)
else:
    matplot.show()