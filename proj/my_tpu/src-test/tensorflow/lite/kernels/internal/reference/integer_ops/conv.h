/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>
#include <stdio.h>
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "./fully_connected.h"

#include "cfu.h"
#include "perf.h"

namespace tflite {
namespace reference_integer_ops {

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {

  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  //const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  //const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  uint8_t input_matrix[4096][4096];
  int8_t weight_matrix[4096][4096];
  printf("Entering loop at: %u\n", perf_get_mcycle());
  for (int out_y=0; out_y < output_height; ++out_y){
    const int in_y_origin = (out_y * stride_height) - pad_height;
    for (int out_x=0; out_x < output_width; ++out_x){
      const int in_x_origin = (out_x * stride_width) - pad_width;
      for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel){
        for (int filter_y = 0; filter_y < filter_height; ++filter_y){
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          for (int filter_x = 0; filter_x < filter_width; ++filter_x){
            const int in_x = in_x_origin + dilation_width_factor * filter_x;
	    
	    // Zero padding by omitting the areas outside the image.
	    const bool is_point_inside_image = 
		    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) && (in_y < input_height);
	    if(!is_point_inside_image){
	      input_matrix[in_channel*filter_height*filter_width+filter_y*filter_width+filter_x][out_y*output_width+out_x]
		= 0;
	    }
	    else{
              input_matrix[in_channel*filter_height*filter_width+filter_y*filter_width+filter_x][out_y*output_width+out_x]
                = input_data[Offset(input_shape, 0, in_y, in_x, in_channel)] + input_offset;
	    }
          }
        }
      }
    }
  }
  for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel){
    for (int filter_y = 0; filter_y < filter_height; ++filter_y){
        for (int filter_x = 0; filter_x < filter_width; ++filter_x){
            for(int out_channel = 0; out_channel < output_depth; ++out_channel){
                weight_matrix[in_channel*filter_height*filter_width+filter_y*filter_width+filter_x][out_channel]
                    = filter_data[Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)];
            }
        }
    }
  }

  int32_t res[5][5];
  //int32_t mine_check[48];
  for (int batch = 0; batch < batches; ++batch) {
    for(int out=0; out*4 < output_height*output_width; ++out){
      for (int fil=0; fil*4 < output_depth; ++fil){
	for(int j = 0; j < filter_height*filter_width*filter_input_depth; ++j){
	  uint8_t input0 = input_matrix[j][out*4];
	  uint8_t input1 = (out*4+1 < output_height*output_width) ? input_matrix[j][out*4+1] : 0;
	  uint8_t input2 = (out*4+2 < output_height*output_width) ? input_matrix[j][out*4+2] : 0;
	  uint8_t input3 = (out*4+3 < output_height*output_width) ? input_matrix[j][out*4+3] : 0;
	  uint32_t inputs = (input0<<24)|(input1<<16)|(input2<<8)|(input3);
	  uint8_t weight0 = weight_matrix[j][fil*4];
	  uint8_t weight1 = (fil*4+1 < output_depth) ? weight_matrix[j][fil*4+1] : 0;
	  uint8_t weight2 = (fil*4+2 < output_depth) ? weight_matrix[j][fil*4+2] : 0;
	  uint8_t weight3 = (fil*4+3 < output_depth) ? weight_matrix[j][fil*4+3] : 0;
      	  uint32_t weights = (weight0<<24)|(weight1<<16)|(weight2<<8)|(weight3);
	  //if(out==0 && input_width==48){
	  //  printf("%x %x\n ", (unsigned int)inputs, (unsigned int)weights);
	  //}
	  cfu_op1(0, inputs, weights);
	}
	int KM = ((filter_height*filter_width*filter_input_depth)<<16)|(4);
	int N = 4;
	cfu_op3(0, KM, N);
	for(int l=0; l<4 && out*4+l<output_height*output_width; l++){
	  for(int m=0; m<4 && fil*4+m<output_depth; m++){
	    //res[4][4] += 1;
            res[l][m] = cfu_op2(0, l, m);
	    res[l][m] = cfu_op2(0, l, m);
	    //if(out*4+l<48 && fil*4+m==0){
	    //  mine_check[out*4+l] = res[l][m];
	    //}
            if (bias_data) {
              res[l][m] += bias_data[fil*4+m];
            }
            res[l][m] = MultiplyByQuantizedMultiplier(
                res[l][m], output_multiplier[fil*4+m], output_shift[fil*4+m]);
            res[l][m] += output_offset;
            res[l][m] = std::max(res[l][m], output_activation_min);
            res[l][m] = std::min(res[l][m], output_activation_max);
            output_data[Offset(output_shape, batch, (out*4+l)/output_width, (out*4+l)%output_width, fil*4+m)] =
	        static_cast<int8_t>(res[l][m]);
	  }
	}
      }
    }
  }

  /*if(input_width==48){
    for(int i=0;i<10;i++){
      for(int j=0;j<filter_height*filter_width*filter_input_depth;j++){
        printf("%x, ", (unsigned int)input_matrix[j][i]);
      }
      printf("\n");
    }
    printf("\n======================================================================================================\n");
  }*/


/*
  printf("\n\n\n\n");
  if(input_width==48){
    for(int i=0;i<input_depth;i++){
      for(int j=0;j<input_width;j++){
        printf("%d, ", input_check[i][j]);
      }
      printf("\n");
    }

    for(int i=0;i<input_depth;i++){
      printf("%d  ", weight_check[0][i]);
    }
    printf("\nmine: ");
    for(int i=0;i<48;i++){
      printf("%ld ", mine_check[i]);
    }
    printf("\n");
  }*/
  printf("Exit loop at: %u\n", perf_get_mcycle());
  
}

inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
