PlotBitrate
===========

FFProbe Bitrate Graph

This project contains a script for plotting the bitrate of an audio or video
stream over time. To get the frame bitrate data ffprobe is used from the
ffmpeg package. The ffprobe data is streamed to python as xml frame metadata
and optionaly sorted by frame type. Matplotlib is used to plot the overall bitrate
or each frame type on the same graph with lines for the peak and mean bitrates. 
The resulting bitrate graph can be saved as an image.

Possible outputs are:
* Image types (png, svg, pdf, ...)
* Raw frame data (csv, xml)

Requirements:

* Python >= 3.6
* FFMpeg >= 1.2 with the ffprobe command
* Matplotlib python 3 library (install: `python3 -m pip install -U --user matplotlib`)


Usefull Options
==============

The raw frame data can be stored in an xml file with the option `-f xml_raw`,
which the graph can be plotted from.
This is useful if the graph is should be shown multiple times with different options,
as the source file doesn't need to be scanned again.

The option `--downscale` (or `-d`) is usefull if the video is very long and an overview
of the bitrate fluctuation is sufficient and zooming in is not necessary.
This behavior resembles that of the tool "Bitrate Viewer".
With this option, videos will be shown as a downscaled graph, meaning not every second is being displayed.
Multiple seconds will be grouped together and the max value will be drawn.
This downscaling is not applied when viewing individual frame types as this woud lead to wrong graphs.
This behavior cann be adjusted with the `--max-display-values` option.
The default value is 700, meaning that at most 700 individual seconds/bars are drawn.


Usage Examples
==============

Show video stream bitrate in a window with progress.

```
./plotbitrate.py input.mkv
```

Show downscaled video stream bitrate in a window.

```
./plotbitrate.py -d input.mkv
```

Show video stream bitrate for each frame type in a window.

```
./plotbitrate.py -t input.mkv
```

Save video stream bitrate to an SVG file.

```
./plotbitrate.py -o output.svg input.mkv
```

Show audio stream bitrate in a window.

```
./plotbitrate.py -s audio input.mkv
```

Save raw ffproble frame data as xml file.

```
./plotbitrate.py -f xml_raw -o frames.xml input.mkv
```

Show bitrate graph from raw xml.

```
./plotbitrate.py frames.xml
```