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

__version__ = "1.0.1"

import argparse
import csv
import datetime
import math
import multiprocessing
import operator
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Union, List, IO, Iterable, Optional, Dict, Tuple

# prefer C-based ElementTree
try:
    import xml.etree.cElementTree as eTree
except ImportError:
    import xml.etree.ElementTree as eTree  # type: ignore

# check for matplot lib
try:
    import matplotlib.pyplot as matplot  # type: ignore
    import matplotlib  # type: ignore
except ImportError:
    sys.exit("Error: Missing package 'python3-matplotlib'")

# check for ffprobe in path
if not shutil.which("ffprobe"):
    sys.exit("Error: Missing ffprobe from package 'ffmpeg'")


@dataclass
class Frame:
    __slots__ = ["time", "size_kbit", "pict_type"]
    time: float
    size_kbit: int
    pict_type: str


class Color(Enum):
    I = "red"
    P = "green"
    B = "blue"
    AUDIO = "indianred"
    FRAME = "indianred"


def parse_arguments() -> argparse.Namespace:
    # get list of supported matplotlib formats
    format_list = list(
        matplotlib.figure.Figure().canvas.get_supported_filetypes().keys()
    )
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
    """ Opens an ffprobe process that reads the format data
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


def open_ffprobe_get_frames(file_path: str,
                            stream_selector: str) -> subprocess.Popen:
    """ Opens an ffprobe process that reads all frame data for
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


def save_raw_xml(file_path: str, target_path: str, stream_selector: str,
                 no_progress: bool) -> None:
    """ Reads all raw frame data from file_path
    and saves it to target_path. """
    if not no_progress:
        last_percent = 0
        with open_ffprobe_get_format(file_path) as proc_format:
            media_time_in_seconds = parse_media_duration(
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
                 file_path
                 ],
                stdout=subprocess.PIPE) as p:
            # start process and iterate over output lines
            for line in p.stdout:
                f.write(line)

                # for progress
                # look for lines starting with frame tag
                # try parsing the time from them and print percent
                if not no_progress \
                        and media_time_in_seconds > 0 \
                        and line.lstrip().startswith(b"<frame "):
                    frame_time = try_get_frame_time_from_node(
                        eTree.fromstring(line))

                    if frame_time is not None:
                        percent = math.floor((frame_time /
                                              media_time_in_seconds) * 100.0)
                    else:
                        percent = 0

                    if percent > last_percent:
                        print_progress(percent)
                        last_percent = percent
            if not no_progress:
                print(flush=True)


def save_raw_csv(raw_frames: List[Frame], target_path: str) -> None:
    """ Saves raw_frames as a csv file. """
    if len(raw_frames) == 0:
        return
    with open(target_path, "w") as f:
        wr = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        wr.writerow(vars(raw_frames[0]).keys())
        wr.writerows((vars(frame).values() for frame in raw_frames))


def parse_media_duration(source: Union[str, IO]) -> float:
    """ Parses the source and returns the extracted total duration. """
    format_data = eTree.parse(source)
    format_elem = format_data.find(".//format")
    duration_str = format_elem.get("duration") \
        if format_elem is not None else None
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


def print_progress(percent: float) -> None:
    sys.stdout.write("\rProgress: {:2}%".format(percent))


def frame_elements(source_iterable: Iterable) -> Iterable[eTree.Element]:
    for _, node in source_iterable:
        if node.tag == "frame":
            yield node


def sum_size(frames: Iterable[Frame]) -> int:
    return sum(frame.size_kbit for frame in frames)


def read_frame_data(source_iterable: Iterable,
                    frame_read_callback: Optional[Callable[[Frame], None]]
                    ) -> List[Frame]:
    """ Iterates over source_iterable and creates a list of Frame objects.
    Will call frame_read_callback on the end of each loop so that progress
    can be calculated.
    """
    data = []
    for node in frame_elements(source_iterable):
        time = try_get_frame_time_from_node(node)
        size = node.get("pkt_size")
        pict_type = node.get("pict_type")
        frame = Frame(
            time=time if time else 0,
            size_kbit=int((float(size) if size else 0) * 8 / 1000),
            pict_type=pict_type if pict_type else "?")
        data.append(frame)
        if frame_read_callback is not None:
            frame_read_callback(frame)
    return data


def group_frames_to_seconds(frames: List[Frame], seconds_start: int,
                            seconds_end: int, seconds_step: int = 1
                            ) -> Dict[int, int]:
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
    seconds_with_frames: Dict[int, List[Frame]] = {}
    for frame in frames:
        seconds_with_frames.setdefault(math.floor(frame.time), []) \
            .append(frame)

    # iterate over seconds with the given step
    mapped_data: Dict[int, int] = {}
    for second in range(seconds_start, seconds_end, seconds_step):
        # if steps is greater than one,
        # this will run over each second in between
        second_sub_step = second
        while second_sub_step < second + seconds_step:
            # take the highest bitrate
            mapped_data[second] = max(
                mapped_data.get(second, 0),
                sum_size(seconds_with_frames.get(second_sub_step, ())))
            second_sub_step += 1
    return mapped_data


def prepare_matplot(window_title: str, total_media_time_in_seconds: int,
                    min_y: Optional[int], max_y: Optional[int]
                    ) -> None:
    """ Prepares the chart and sets up a new figure """

    matplot.figure(figsize=[10, 4]).canvas.set_window_title(window_title)
    matplot.title("Stream Bitrate over Time")
    matplot.xlabel("Time")
    matplot.ylabel("Bitrate (kbit/s)")
    matplot.grid(True, axis="y")
    matplot.tight_layout()

    # set 10 x axes ticks
    matplot.xticks(range(
        0,
        total_media_time_in_seconds + 1,
        max(total_media_time_in_seconds // 10, 1)))

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


def add_frames_as_stacked_bars(
        frames: List[Frame], media_duration_in_s: int
) -> Tuple[int, int, Dict[str, matplotlib.container.BarContainer]]:
    """ Calculates the bitrate for each frame type
    and adds a stacking bar for each
    """
    bars = {}
    sums_of_values: List[int] = []

    # calculate bitrate for each frame type
    # and add a stacking bar for each
    for frame_type in ["I", "B", "P", "?"]:
        filtered_frames = list(filter(lambda x: x.pict_type == frame_type,
                                      frames))
        if len(filtered_frames) == 0:
            continue

        # simple frame grouping to seconds
        seconds_with_bitrates = group_frames_to_seconds(
            filtered_frames, 0, media_duration_in_s)
        bitrate_values = list(seconds_with_bitrates.values())
        bars[frame_type] = matplot.bar(
            seconds_with_bitrates.keys(),
            bitrate_values,
            bottom=sums_of_values if len(sums_of_values) > 0 else 0,
            color=Color[frame_type].value if frame_type in dir(Color)
            else Color.FRAME.value,
            width=1)

        # add current frame bitrate values to all previous
        # needed so that the stacking bars know their min value
        # and so that the peak and mean can be calculated
        if len(sums_of_values) == 0:
            sums_of_values = bitrate_values
        else:
            sums_of_values = list(
                map(operator.add, sums_of_values, bitrate_values))

    return max(sums_of_values), int(statistics.mean(sums_of_values)), bars


def add_frames_as_bar(
        frames: List[Frame], media_duration_in_s: int,
        downscale: bool, max_display_values: int, stream_type: str
) -> Tuple[int, int]:
    second_steps = 1
    if downscale:
        # calculate how many seconds in between should be left out
        # so that only a maximum of 'args.max_display_values'
        # number of values are shown
        second_steps = max(1, media_duration_in_s // max_display_values) \
            if max_display_values >= 1 else 1

    seconds_with_bitrates = group_frames_to_seconds(
        frames, 0, media_duration_in_s, second_steps)

    if stream_type == "audio":
        color = Color.AUDIO.value
    elif stream_type == "video":
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
    all_values = list(
        group_frames_to_seconds(frames, 0, media_duration_in_s).values())
    return max(all_values), int(statistics.mean(all_values))


def draw_horizontal_line_with_text(pos_y: int, pos_h_percent: float, text: str):
    # calculate line position (above line)
    text_x = matplot.xlim()[1] * pos_h_percent
    text_y = pos_y + ((matplot.ylim()[1] - matplot.ylim()[0]) * 0.015)

    # draw as think black line with text
    matplot.axhline(pos_y, linewidth=1.5, color="black")
    matplot.text(text_x, text_y, text,
                 horizontalalignment="center", fontweight="bold",
                 color="black")


def main():
    args = parse_arguments()

    # if the output is raw xml, just call the function and exit
    if args.format == "xml_raw":
        save_raw_xml(args.input, args.output, args.stream_spec,
                     args.no_progress)
        sys.exit(0)

    media_duration_in_s = 0
    source_is_xml = args.input.endswith(".xml")

    # read total time from format
    if source_is_xml:
        media_duration_in_s = int(parse_media_duration(args.input))
    else:
        process = open_ffprobe_get_format(args.input)
        media_duration_in_s = int(parse_media_duration(process.stdout))

    if media_duration_in_s == 0:
        sys.exit("Error: Failed to determine stream duration")

    # open frame data reader for media or xml file
    if source_is_xml:
        frames_source = eTree.iterparse(args.input)
    else:
        proc_frame = open_ffprobe_get_frames(args.input, args.stream_spec)
        frames_source = eTree.iterparse(proc_frame.stdout)

    # only report progress if it changed
    progress_last_percent = 0

    def report_frame_progress(frame: Frame):
        nonlocal progress_last_percent
        percent = math.floor((frame.time / media_duration_in_s) * 100.0)
        if percent > progress_last_percent:
            print_progress(percent)
            progress_last_percent = percent

    # read frame data
    frames_raw = read_frame_data(
        frames_source,
        report_frame_progress if not args.no_progress else None)

    if not args.no_progress:
        print(flush=True)

    # check for success
    if not frames_raw:
        sys.exit("Error: No frame data, failed to execute ffprobe")

    # if the output is csv raw, write the file and we're done
    if args.format == "csv_raw":
        save_raw_csv(frames_raw, args.output)
        sys.exit(0)

    prepare_matplot(args.input, media_duration_in_s, args.min, args.max)

    bars: Dict[str, matplotlib.container.BarContainer] = {}
    if args.show_frame_types and args.stream == "video":
        peak, mean, bars = add_frames_as_stacked_bars(frames_raw,
                                                      media_duration_in_s)
    else:
        peak, mean = add_frames_as_bar(frames_raw,
                                       media_duration_in_s,
                                       args.downscale,
                                       args.max_display_values,
                                       args.stream)

    draw_horizontal_line_with_text(
        pos_y=peak,
        pos_h_percent=0.08,
        text="peak ({:,})".format(peak))
    draw_horizontal_line_with_text(
        pos_y=mean,
        pos_h_percent=0.92,
        text="mean ({:,})".format(mean))

    if bars:
        matplot.legend(bars.values(), bars.keys())

    # render graph to file (if requested) or screen
    if args.output:
        matplot.savefig(args.output, format=args.format, dpi=300)
    else:
        matplot.show()


if __name__ == "__main__":
    main()
