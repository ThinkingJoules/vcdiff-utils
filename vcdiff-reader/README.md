# vcdiff-reader

## Overview
The `vcdiff-reader` library is used to help read a raw VCDIFF patch format into Rust structs.

## Features
- Performs robust validation to ensure correct VCDIFF format adherence.
- Detects implicit sequences and makes them explicit
- Works through a patch file in a window and instruction oriented step-wise procedure.
