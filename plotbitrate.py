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

__version__ = "1.1.0.a1"

import argparse
import csv
import datetime
import math
import multiprocessing
import shutil
import statistics
import subprocess
import sys
import inspect
from collections import OrderedDict
from enum import Enum
from typing import Callable, Union, List, IO, Iterable, Optional, Dict, Tuple, \
    Generator
    
# prefer C-based ElementTree
try:
    import xml.etree.cElementTree as eTree
except ImportError:
    import xml.etree.ElementTree as eTree  # type: ignore

# check for matplot lib
try:
    import matplotlib # type: ignore
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as matplot # type: ignore
except ImportError:
    sys.exit("Error: Missing package 'python3-matplotlib'")

# check for ffprobe in path
if not shutil.which("ffprobe"):
    sys.exit("Error: Missing ffprobe from package 'ffmpeg'")

if sys.version_info >= (3, 6):
    # if python 3.6 or greater: use dataclass version of Frame
    import dataclasses
    from frame_dataclass import Frame
else:
    # use normal class version of Frame
    class Frame:
        def __init__(self, time, size, pict_type):
            self.time = time
            self.size = size
            self.pict_type = pict_type
        
        @staticmethod
        def get_fields():
            return ['time', 'size', 'pict_type']

class Color(Enum):
    I = "red"
    P = "green"
    B = "blue"
    AUDIO = "C2"
    FRAME = "C0"


def parse_arguments() -> argparse.Namespace:
    """ Parses all arguments and returns them as an object. """
    
    if sys.version_info >= (3, 6):
        supported_filetypes = matplotlib.figure.Figure().canvas\
            .get_supported_filetypes().keys()
    else:
        fig = matplot.figure()
        supported_filetypes = fig.canvas.get_supported_filetypes().keys()
        matplot.close(fig)

    # get list of supported matplotlib formats
    format_list = list(supported_filetypes)
    
    format_list.append("xml_raw")
    format_list.append("csv_raw")

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Graph bitrate for audio/video stream")
    parser.add_argument("input", help="input file/stream", metavar="INPUT")
    parser.add_argument("-s", "--stream", help="Stream type (default: video)",
                        choices=["audio", "video"], default="video")
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("-f", "--format", help="Output file format",
                        choices=format_list)
    parser.add_argument("--no-progress", help="Hides progress",
                        action="store_true")
    parser.add_argument("--min", help="Set plot minimum (kbps)", type=int)
    parser.add_argument("--max", help="Set plot maximum (kbps)", type=int)
    parser.add_argument("-t", "--show-frame-types",
                        help="Show bitrate of different frame types",
                        action="store_true")
    parser.add_argument(
        "-d",
        "--downscale",
        help="Enable downscaling of values, so that the visible" +
             "level of detail in the graph is reduced and rendered faster. " +
             "This is useful if the video is very long and an overview " +
             "of the bitrate fluctuation is sufficient.",
        action="store_true")
    parser.add_argument(
        "--max-display-values",
        help="If downscaling is enabled, set the maximum number of values " +
             "shown on the x axis. Will downscale if video length is longer " +
             "than the given value. Will not downscale if set to -1. " +
             "Not compatible with option --show-frame-types (default: 700)",
        type=int,
        default=700)
    arguments = parser.parse_args()

    # check if format given without output file
    if arguments.format and not arguments.output:
        sys.exit("Error: Output format requires output file")

    # check given y-axis limits
    if arguments.min and arguments.max and (arguments.min >= arguments.max):
        sys.exit("Error: Maximum should be greater than minimum")

    arguments_dict = vars(arguments)

    # set ffprobe stream specifier
    if arguments.stream == "audio":
        arguments_dict["stream_spec"] = "a"
    elif arguments.stream == "video":
        arguments_dict["stream_spec"] = "V"
    else:
        raise RuntimeError("Invalid stream type")

    return arguments


def open_ffprobe_get_format(file_path: str) -> subprocess.Popen:
    """
    Opens an ffprobe process that reads the format data
    for file_path and returns the process.
    """
    return subprocess.Popen(
        ["ffprobe",
         "-hide_banner",
         "-loglevel", "error",
         "-show_entries", "format",
         "-print_format", "xml",
         file_path
         ],
        stdout=subprocess.PIPE)


def open_ffprobe_get_frames(
        file_path: str,
        stream_selector: str
) -> subprocess.Popen:
    """
    Opens an ffprobe process that reads all frame data for
    file_path and returns the process.
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
         file_path
         ],
        stdout=subprocess.PIPE)


def save_raw_xml(
        file_path: str,
        target_path: str,
        stream_selector: str,
        no_progress: bool
) -> None:
    """
    Reads all raw frame data from file_path
    and saves it to target_path.
    """
    if not no_progress:
        last_percent = 0
        with open_ffprobe_get_format(file_path) as proc_format:
            assert proc_format.stdout is not None
            duration = parse_media_duration(proc_format.stdout)

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
                 file_path
                 ],
                stdout=subprocess.PIPE) as p:
            assert p.stdout is not None
            # start process and iterate over output lines
            for line in p.stdout:
                f.write(line)

                # for progress
                # look for lines starting with frame tag
                # try parsing the time from them and print percent
                if not no_progress \
                        and duration > 0 \
                        and line.lstrip().startswith(b"<frame "):
                    frame_time = \
                        try_get_frame_time_from_node(eTree.fromstring(line))

                    if frame_time is not None:
                        percent = math.floor((frame_time / duration) * 100.0)
                    else:
                        percent = 0

                    if percent > last_percent:
                        print_progress(percent)
                        last_percent = percent
            if not no_progress:
                print(flush=True)


def save_raw_csv(raw_frames: Iterable[Frame], target_path: str) -> None:
    """ Saves raw_frames as a csv file. """
    if sys.version_info >= (3, 6):
        fields = [f.name for f in dataclasses.fields(Frame)]
    else:
        fields = Frame.get_fields()

    with open(target_path, "w") as file:
        wr = csv.writer(file, quoting=csv.QUOTE_NONE)
        wr.writerow(fields)
        for frame in raw_frames:
            wr.writerow(getattr(frame, field) for field in fields)


def media_duration(source: str) -> float:
    if source.endswith(".xml"):
        return parse_media_duration(source)
    else:
        with open_ffprobe_get_format(source) as process:
            assert process.stdout is not None
            return parse_media_duration(process.stdout)


def parse_media_duration(source: Union[str, IO]) -> float:
    """ Parses the source and returns the extracted total duration. """
    format_data = eTree.parse(source)
    format_elem = format_data.find(".//format")
    duration_str = \
        format_elem.get("duration") if format_elem is not None else None
    return float(duration_str) if duration_str is not None else 0


def try_get_frame_time_from_node(node: eTree.Element) -> Optional[float]:
    for attribute_name in ["best_effort_timestamp_time", "pkt_pts_time"]:
        value = node.get(attribute_name)
        if value is not None:
            try:
                return float(value)
            except ValueError:
                continue
    return None


def create_progress(duration: int):
    last_percent = 0

    def report_progress(frame: Optional[Frame]):
        nonlocal last_percent
        if frame:
            percent = math.floor((frame.time / duration) * 100.0)
            if percent > last_percent:
                print_progress(percent)
                last_percent = percent
        else:
            last_percent = 100
            print_progress(last_percent)
            print()

    return report_progress


def print_progress(percent: float) -> None:
    print("Progress: {:2}%".format(percent), end="\r")


def frame_elements(source_iterable: Iterable) -> Iterable[eTree.Element]:
    for _, node in source_iterable:
        if node.tag == "frame":
            yield node


def read_frame_data_gen(
        source: str,
        stream_spec: str,
        frame_progress_func: Optional[Callable[[Optional[Frame]], None]]
) -> Generator[Frame, None, None]:
    source_iter = "" # type: Union[str, IO]
    if source.endswith(".xml"):
        source_iter = source
    else:
        proc = open_ffprobe_get_frames(source, stream_spec)
        assert proc.stdout is not None
        source_iter = proc.stdout

    for f in read_frame_data_gen_internal(source_iter):
        if frame_progress_func:
            frame_progress_func(f)
        yield f

    if frame_progress_func:
        frame_progress_func(None)


def read_frame_data_gen_internal(
        source: Union[str, IO]
) -> Generator[Frame, None, None]:
    """
    Creates an iterator from source_iterable and yields Frame objects.
    """
    for node in frame_elements(eTree.iterparse(source)):
        # get data
        time = try_get_frame_time_from_node(node)
        size = node.get("pkt_size")
        pict_type = node.get("pict_type")
        # clear node to free parsed data
        node.clear()

        # construct and append frame
        yield Frame(
            time=time if time else 0,
            size=int(size) if size else 0,
            pict_type=pict_type if pict_type else "?"
        )


def frames_to_kbits(
        frames: Iterable[Frame],
        seconds_start: int,
        seconds_end: int
) -> Generator[Tuple[int, int], None, None]:
    """
    Creates a generator yielding every second between seconds_start
    and seconds_end (including both) and its summed size in kbit.

    The frames iterable must be sorted by frame time.
    """
    frames_iter = iter(frames)
    last_frame_second = 0
    last_frame_size = 0

    # loop over every second
    for second in range(seconds_start, seconds_end + 1):

        # restore size of a saved frame from last iteration
        # if it's for the current second
        if last_frame_second == second:
            size = last_frame_size
        else:
            size = 0

        # advance iterator only if the saved frame data
        # is not for a future second
        if last_frame_second <= second:
            # advances the iterator until it's at a frame
            # belonging to a future second
            for frame in frames_iter:
                frame_second = math.floor(frame.time)
                if frame_second < second:
                    continue
                elif frame_second == second:
                    # frame is current second, so sum up
                    size += frame.size
                else:
                    # current frame is not in current second
                    # store its size and second and break iteration
                    last_frame_second = frame_second
                    last_frame_size = frame.size
                    break

        yield second, int(size * 8 / 1000)


def downscale_bitrate(
        bitrates: Dict[int, int],
        factor: int
) -> Generator[Tuple[int, int], None, None]:
    """
    Groups bitrates together and takes the highest bitrate as the value.

    Args:
        bitrates: dict containing seconds with bitrates
        factor: which seconds to keep (1 is every, 2 is every other and so on)

    Example:

        given the parameters:

        bitrates = {
            0: 3400,
            1: 5290
            2: 4999
            3: 7500
            4: 0
            5: 7800
            6: 3000
        }
        factor = 3

        this function will return an iterator giving:
        (0, 5290)
        (3, 7800)
        (6, 3000)
    """
    # iterate over all seconds to yield
    for second in range(min(bitrates.keys()), max(bitrates.keys()) + 1, factor):
        # iterate over all seconds in between
        # and find the highest bitrate
        max_b = max(bitrates.get(s, 0) for s in range(second, second + factor))
        yield second, max_b


def prepare_matplot(
        window_title: str,
        duration: int,
        min_y: Optional[int],
        max_y: Optional[int]
) -> None:
    """ Prepares the chart and sets up a new figure """

    matplot.figure(figsize=[10, 4]).canvas.set_window_title(window_title)
    matplot.title("Stream Bitrate over Time")
    matplot.xlabel("Time")
    matplot.ylabel("Bitrate (kbit/s)")
    matplot.grid(True, axis="y")

    # set 10 x axes ticks
    matplot.xticks(range(0, duration + 1, max(duration // 10, 1)))

    # format axes values
    matplot.gcf().axes[0].xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda x, loc: datetime.timedelta(seconds=int(x))))
    matplot.gca().get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda x, loc: "{:,}".format(int(x))))

    # set y-axis limits if requested
    if min_y:
        matplot.ylim(ymin=min_y)
    if max_y:
        matplot.ylim(ymax=max_y)


def add_stacked_areas(
        frames: Iterable[Frame],
        duration: int
) -> Tuple[int, int, Dict[str, matplotlib.collections.PolyCollection]]:
    """ Calculates the bitrate for each frame type
    and adds a stacking bar for each
    """
    bars = {}
    sums_of_values = [] # type: List[int]
    frames_list = frames if isinstance(frames, list) else list(frames)

    # calculate bitrate for each frame type
    # and add a stacking bar for each
    for frame_type in ["I", "B", "P", "?"]:
        filtered_frames = [f for f in frames_list if f.pict_type == frame_type]
        if len(filtered_frames) == 0:
            continue

        bitrates = OrderedDict(frames_to_kbits(filtered_frames, 0, duration))
        seconds = list(bitrates.keys())
        values = list(bitrates.values())

        if len(sums_of_values) == 0:
            values_min = [0]
            values_max = values
        else:
            values_min = sums_of_values
            values_max = [
                sum(pair) for pair in zip(sums_of_values, values)
            ]
        sums_of_values = values_max
        color = Color[frame_type].value if frame_type in dir(Color) \
            else Color.FRAME.value
        bars[frame_type] = matplot.fill_between(
            seconds, values_min, values_max, linewidth=0.5, color=color
        )

    return max(sums_of_values), int(statistics.mean(sums_of_values)), bars


def add_area(
        frames: Iterable[Frame],
        duration: int,
        downscale: bool,
        max_display_values: int,
        stream_type: str
) -> Tuple[int, int]:
    bitrates = OrderedDict(frames_to_kbits(frames, 0, duration))
    bitrate_max = max(bitrates.values())
    bitrate_mean = int(statistics.mean(bitrates.values()))

    if downscale and 0 < max_display_values < duration:
        factor = duration // max_display_values
        bitrates = OrderedDict(downscale_bitrate(bitrates, factor))

    seconds = list(bitrates.keys())
    values = list(bitrates.values())
    color = Color.AUDIO.value if stream_type == "audio" else Color.FRAME.value
    matplot.plot(seconds, values, linewidth=0.5, color=color)
    matplot.fill_between(seconds, 0, values, linewidth=0.5, color=color)
    return bitrate_max, bitrate_mean


def draw_horizontal_line_with_text(
        pos_y: int,
        pos_h_percent: float,
        text: str
) -> None:
    # calculate line position (above line)
    text_x = matplot.xlim()[1] * pos_h_percent
    text_y = pos_y + ((matplot.ylim()[1] - matplot.ylim()[0]) * 0.015)

    # draw as think black line with text
    matplot.axhline(pos_y, linewidth=1.5, color="black")
    matplot.text(
        text_x, text_y, text,
        horizontalalignment="center", fontweight="bold", color="black"
    )


def main():
    args = parse_arguments()

    # if the output is raw xml, just call the function and exit
    if args.format == "xml_raw":
        save_raw_xml(
            args.input, args.output, args.stream_spec, args.no_progress
        )
        sys.exit(0)

    duration = math.floor(media_duration(args.input))
    if duration == 0:
        sys.exit("Error: Failed to determine stream duration")

    progress_func = create_progress(duration) if not args.no_progress else None
    frames = read_frame_data_gen(
        args.input, args.stream_spec, progress_func
    )

    # if the output is csv raw, write the file and we're done
    if args.format == "csv_raw":
        save_raw_csv(frames, args.output)
        sys.exit(0)

    prepare_matplot(args.input, duration, args.min, args.max)

    legend = None
    if args.show_frame_types and args.stream == "video":
        peak, mean, legend = add_stacked_areas(frames, duration)
    else:
        peak, mean = add_area(
            frames, duration, args.downscale, args.max_display_values,
            args.stream
        )

    draw_horizontal_line_with_text(
        pos_y=peak,
        pos_h_percent=0.08,
        text="peak ({:,})".format(peak)
    )
    draw_horizontal_line_with_text(
        pos_y=mean,
        pos_h_percent=0.92,
        text="mean ({:,})".format(mean)
    )

    if legend:
        matplot.legend(legend.values(), legend.keys(), loc="upper right")

    # render graph to file (if requested) or screen
    if args.output:
        matplot.savefig(args.output, format=args.format, dpi=300)
    else:
        matplot.tight_layout()
        matplot.show()


if __name__ == "__main__":
    main()
