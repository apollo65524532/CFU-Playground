#!/bin/env python
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This variable lists symbols to define to the C preprocessor
export DEFINES := TF_LITE_SHOW_MEMORY_USE

# Conditionally enable tail capture
ifdef TAIL_CAPTURE
DEFINES += TAIL_ROM_CAPTURE
else
DEFINES += TAIL_ROM_USE
endif

# Uncomment this line to use software defined CFU functions in software_cfu.cc
#DEFINES += CFU_SOFTWARE_DEFINED

# Uncomment this line to skip debug code (large effect on performance)
DEFINES += NDEBUG

# Uncomment this line to skip individual profiling output (has minor effect on performance).
DEFINES += NPROFILE

# Uncomment to include pdti8 in built binary
DEFINES += INCLUDE_MODEL_PDTI8

# Uncomment to include micro_speech in built binary
DEFINES += INCLUDE_MODEL_MICRO_SPEECH

# Uncomment to include magic_wand in built binary
#DEFINES += INCLUDE_MODEL_MAGIC_WAND

# Uncomment to include mnv2 in built binary (adds ~500kB to binary)
#DEFINES += INCLUDE_MODEL_MNV2

# Uncomment to include  mlcommons tiny v0.1 v0.1 anomaly detection in built binary
#DEFINES += INCLUDE_MODEL_MLCOMMONS_TINY_V01_ANOMD

# Uncomment to include  mlcommons tiny v0.1 v0.1 image classification in built binary
#DEFINES += INCLUDE_MODEL_MLCOMMONS_TINY_V01_IMGC

# Uncomment to include  mlcommons tiny v0.1 v0.1 keyword spotting in built binary
#DEFINES += INCLUDE_MODEL_MLCOMMONS_TINY_V01_KWS

# Uncomment to include  mlcommons tiny v0.1 v0.1 visual wake words in built binary
#DEFINES += INCLUDE_MODEL_MLCOMMONS_TINY_V01_VWW

# Uncomment to include all TFLM examples (pdti8, micro_speech, magic_wand)
#DEFINES += INCLUDE_ALL_TFLM_EXAMPLES

include ../proj.mk
