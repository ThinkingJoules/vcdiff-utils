# vcdiff-testing

This repository is used for testing purposes only.

## Build Instructions

You will want to clone the whole workspace for `vcdiff-utils`.

You will also need to install and build [this lib](https://github.com/ThinkingJoules/open-vcdiff-rs-bindings) locally. Follow its readme.

This uses a fork of one of the xdelta3 bindings, because it allows a more useful api to work with our system.

# Problems
It seems that xdelta3 has an implementation error? It cannot decode a vanilla open-vcdiff encoded patch file. My decoder also gives and error for trying to decode one of its patch files. However, my decoder works with any of the open-vcdiff files.

I'm not an expert coder, so I don't want to say my code is right, but clearly 1 or more of these 3 implementations have errors.

My decoder will work on some xdelta3 files, but it really depends on how it got encoded.