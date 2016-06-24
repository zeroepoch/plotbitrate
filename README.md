PlotBitrate
===========

FFProbe Bitrate Graph

This project contains a script for plotting the bitrate of an audio or video
stream over time.  To get the frame bitrate data ffprobe is used from the
ffmpeg package.  The ffprobe data is streamed to python as xml frame metadata
and sorted by frame type.  Matplotlib is used to plot each frame type on the
same graph with lines for the peak and mean bitrates.  The resulting bitrate
graph can be saved as an image.  Variable framerate streams are not currently
accurately calculated.

Requirements:

* Python 3.x
* FFMpeg >= 1.2 with the ffprobe command
* Matplotlib python 3 library


Usage Examples
==============

Show video stream bitrate in a window

```
./plotbitrate.py input.mkv
```

Save video stream bitrate to an SVG file

```
./plotbitrate.py -o output.svg input.mkv
```

Show audio stream bitrate in a window

```
./plotbitrate.py -s audio input.mkv
```
