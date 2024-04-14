# vcdiff-merger

vcdiff-merger is a library that provides utilities for merging VCDIFF files. VCDIFF (Delta) is a format for encoding differences between two files, commonly used for efficient binary patching.

## Features
Used to create a summary patch between 2 or more patches.

This uses a Merger struct that will allow for early termination. Basically, if a merge patch no longer contains any Copy instructions, merging more patches will have no effect.
