PlotBitrate
===========

FFProbe Bitrate Graph

This project contains a script for plotting the bitrate of an audio or video
stream over time. To get the frame bitrate data ffprobe is used from the ffmpeg
package. The ffprobe data is streamed to python as xml frame metadata and
optionally sorted by frame type. Matplotlib is used to plot the overall bitrate
or each frame type on the same graph with lines for the peak and mean bitrates.
The resulting bitrate graph can be saved as an image.

Possible outputs are:
* Image types (png, svg, pdf, ...)
* Raw frame data (csv, xml)

Requirements:
* Python >= 3.6
* FFmpeg >= 1.2 with the ffprobe command
* Matplotlib
* PyQt5 or PyQt6 (optional for image file output)

For using the script from source, install the requirements with
`pip install -r requirements.txt` or use the `requirements-dev.txt`
for development purposes.

Installation
------------

`pip install plotbitrate`

If you encounter the error message `qt.qpa.plugin: Could not load the Qt
platform plugin "xcb" in "" even though it was found.` while running
`plotbitrate`, then you may need to install your distribution's
`libxcb-cursor0` package. If this package is already installed and/or doesn't
resolve the issue, then you can try using `QT_DEBUG_PLUGINS=1 plotbitrate
input.mkv 2>&1 | grep "No such file"` to determine if other libraries are
missing.

Useful Options
--------------

The raw frame data can be stored in an xml file with the option `-f xml_raw`,
which the graph can be plotted from. This is useful if the graph should be
shown multiple times with different options, as the source file doesn't need to
be scanned again.

The option `--downscale` (or `-d`) is useful if the video is very long and an
overview of the bitrate fluctuation is sufficient and zooming in is not
necessary. This behavior resembles that of the tool "Bitrate Viewer". With this
option, videos will be shown as a downscaled graph, meaning not every second is
being displayed. Multiple seconds will be grouped together and the max value
will be drawn. This downscaling is not applied when viewing individual frame
types as this would lead to wrong graphs. This behavior can be adjusted with
the `--max-display-values` option. The default value is 700, meaning that at
most 700 individual seconds/bars are drawn.

CSV Output
----------

You may find it useful to save the raw frame data to a CSV file so the frame
data can be processed using another tool. This turns `plotbitrate` into more of
a helper tool rather than a visualization tool.

One example may be using `gnuplot` to show an impulse plot for every single
frame split by frame type. Below is an example `gnuplot` script that mimics an
earlier version of `plotbitrate`.

```
#!/usr/bin/gnuplot -persist
set datafile separator ","
plot "< awk -F, '{if($3 == \"I\") print}' frames.csv" u 1:2 t "I" w impulses lt rgb "red", \
     "< awk -F, '{if($3 == \"P\") print}' frames.csv" u 1:2 t "P" w impulses lt rgb "green", \
     "< awk -F, '{if($3 == \"B\") print}' frames.csv" u 1:2 t "B" w impulses lt rgb "blue"
```

The necessary input data can be generated using:

```
plotbitrate -o frames.csv input.mkv
```

Usage Examples
--------------

Show video stream bitrate in a window with progress.

```
plotbitrate input.mkv
```

Show downscaled video stream bitrate in a window.

```
plotbitrate -d input.mkv
```

Show video stream bitrate for each frame type in a window.

```
plotbitrate -t input.mkv
```

Save video stream bitrate to an SVG file.

```
plotbitrate -o output.svg input.mkv
```

Show audio stream bitrate in a window.

```
plotbitrate -s audio input.mkv
```

Save raw ffproble frame data as xml file.

```
plotbitrate -f xml_raw -o frames.xml input.mkv
```

Show bitrate graph from raw xml.

```
plotbitrate frames.xml
```
