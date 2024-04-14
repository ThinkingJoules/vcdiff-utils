# vcdiff-utils: A comprehensive set of Rust libraries for working with the VCDIFF format

# About
The VCDIFF format (RFC 3284) is a versatile and efficient way to represent differences between files. This workspace provides a suite of Rust libraries for creating, reading, manipulating, and applying VCDIFF files.

There are no dependencies as this was implemented from scratch.

# Crates
The vcdiff-utils workspace includes the following crates:

- vcdiff-common
    - Defines core VCDIFF data structures (header, instructions, windows, etc.)
    - Provides traits for interacting with VCDIFF components abstractly, enabling flexibility.
- vcdiff-reader
    - Parses VCDIFF-formatted files into usable Rust data structures.
    - Performs robust validation to ensure correct VCDIFF format adherence.
    - Detects implicit sequences and makes them explicit
- vcdiff-decoder
    - Applies VCDIFF patches to source files for target file reconstruction.
- vcdiff-writer
    - Helps generate valid VCDIFF format files.
    - This is not an encoder. This is the opposite of the vcdiff-reader.
- vcdiff-merger
    - Combines multiple intermediate VCDIFF patches into a single summary patch.
    - Detects and handles potential merge conflicts intelligently.

- vcdiff-testing
    - Only used for testing.
    - Tests against xdelta3 and Google's open-vcdiff lib.

# Assumptions
- Maximum single instruction length is a u32::MAX
- Maximum super string 'U' length is u32::MAX.
