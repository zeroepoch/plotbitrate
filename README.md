PlotBitrate
===========

FFProbe Bitrate Graph

This project contains a script for plotting the bitrate of an audio or video
stream over time.  To get the frame bitrate data ffprobe is used from the
ffmpeg package.  The ffprobe data is streamed to python as xml frame metadata
and optionaly sorted by frame type.  Matplotlib is used to plot the overall bitrate
or each frame type on the same graph with lines for the peak and mean bitrates. 
The resulting bitrate graph can be saved as an image.

By default, the video bitrate is shown and longer videos will be shown as a downscaled graph, as bitrate bars for every second get rather messy and also very slow to draw. This downscaling is not applied when viewing individual frame types as this woud lead to wrong graphs.
If downscaling is not wanted, it can be disabled by providing the option `--max-display-values -1`.
The default value is 700, meaning that at most 700 individual seconds/bars are drawn.
This behavior resembles that of the tool "Bitrate Viewer".

Possible outputs are:
* Image types (png, svg, pdf, ...)
* Raw frame data (csv, xml)

Requirements:

* Python 3.x
* FFMpeg >= 1.2 with the ffprobe command
* Matplotlib python 3 library (install: `python3 -m pip install -U --user matplotlib`)


Usage Examples
==============

Show video stream bitrate in a window with progress.

```
./plotbitrate.py -p input.mkv
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
The option `--progress` is not (yet) available in this case.

```
./plotbitrate.py -f xml_raw -o frames.xml input.mkv
```

Show bitrate graph from raw xml.

```
./plotbitrate.py frames.xml
```