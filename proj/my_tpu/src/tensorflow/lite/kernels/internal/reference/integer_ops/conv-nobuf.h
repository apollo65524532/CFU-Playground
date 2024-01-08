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
#include <stdlib.h>
// #include <vector>
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
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

  const int32_t input_offset = params.input_offset;

/************************************************************************/
  // const int stride_width = params.stride_width;
  // const int stride_height = params.stride_height;
  // const int dilation_width_factor = params.dilation_width_factor;
  // const int dilation_height_factor = params.dilation_height_factor;
  // const int pad_width = params.padding_values.width;
  // const int pad_height = params.padding_values.height;
  // const int batches = MatchingDim(input_shape, 0, output_shape, 0);//I delete
/*************************************************************************/
  const int32_t output_offset = params.output_offset;

  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // const int input_height = input_shape.Dims(1);
  // const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  // const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  // const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
/*****************************************1DDDDDD**************************************************************/
  //  Im2Col dimension
  // const int im2col_height = filter_height * filter_width * input_depth;// size of patch
  // const int im2col_width = output_height * output_width;//number of output elements
  // // const int im2col_size = im2col_height * im2col_width;

  // //Im2Col  data
  // // int8_t* im2col_data = new int8_t[im2col_height * im2col_width]();
  // // printf("%d\n",im2col_height * im2col_width);
  // int8_t im2col_data[19000];
  // for(int i=0;i<19000 ; i++){
  //   im2col_data[i] = 0;
  // }
  // // im2col transition
  // for (int batch = 0; batch < batches; ++batch) {
  //   for (int out_y = 0; out_y < output_height; ++out_y) {
  //     for (int out_x = 0; out_x < output_width; ++out_x) {
  //       for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
  //         for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
  //           for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
  //             const int in_y = (out_y * stride_height +
  //                               dilation_height_factor * filter_y - pad_height);
  //             const int in_x = (out_x * stride_width +
  //                               dilation_width_factor * filter_x - pad_width);

  //             // Zero padding by omitting the areas outside the image.
  //             const bool is_point_inside_image =
  //                 (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
  //                 (in_y < input_height);

  //                 // Calculate the index in the im2col matrix.
  //               const int im2col_row = out_y * output_width + out_x;
  //               const int im2col_col = (filter_y * filter_width * input_depth) +
  //                                     (filter_x * input_depth) + in_channel;
  //             if (!is_point_inside_image) {
  //               im2col_data[im2col_row * im2col_height + im2col_col] = 0;
  //             }else{
                
  //               // Copy the input value to the im2col matrix.
  //               im2col_data[im2col_row * im2col_height + im2col_col] =
  //                   input_data[Offset(input_shape, batch, in_y, in_x, in_channel)];
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  // for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
  //     // auto group = out_channel / filters_per_group;
  //     for (int im2col_row = 0; im2col_row < im2col_width; ++im2col_row) {
  //       int32_t acc = 0;
  //       for (int im2col_col = 0; im2col_col < im2col_height; ++im2col_col) {
  //         int32_t filter_val = filter_data[Offset(
  //             filter_shape, out_channel, im2col_col / (input_depth), im2col_col % (input_depth), 0)];
  //         int32_t input_val = im2col_data[im2col_row * im2col_height + im2col_col];
  //         acc += filter_val * (input_val + input_offset);
  //       }
  //       if (bias_data) {
  //         acc += bias_data[out_channel];
  //       }
  //       acc = MultiplyByQuantizedMultiplier(
  //           acc, output_multiplier[out_channel], output_shift[out_channel]);
  //       acc += output_offset;
  //       acc = std::max(acc, output_activation_min);
  //       acc = std::min(acc, output_activation_max);
  //       output_data[Offset(output_shape, 0, im2col_row / output_width,
  //                          im2col_row % output_width, out_channel)] =
  //           static_cast<int8_t>(acc);
  //     }
  //   }
  // for(int i=0;i<im2col_height*im2col_width;i++){
  //   printf("%d\n",output_data[i]);
  // }
  /****************************1DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD*****************************************************/
// Determine the dimensions of the 2D matrices.
const int matrix_a_rows = output_height * output_width;
const int matrix_a_cols = filter_height * filter_width * input_depth;
// const int matrix_b_rows = filter_height * filter_width * input_depth;
const int matrix_b_cols = output_depth;

// printf("no buffer convolution\n");

// Initialize matrices for matrix multiplication.
uint8_t matrix_a[4096][4096]; //uint8_t -> int8_t
int8_t matrix_b[4096][4096];
int32_t output_matrix[4096][4096]; //uint32_t 0> int32_t

// printf("enter at : %u\n",perf_get_mcycle());
perf_enable_counter(0);
// Fill matrix_a with values from input_data.
for (int out_y = 0; out_y < output_height; ++out_y) {
    for (int out_x = 0; out_x < output_width; ++out_x) {
        int matrix_a_row = out_y * output_width + out_x;
        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                    int matrix_a_col = (filter_y * filter_width * input_depth) +
                                       (filter_x * input_depth) + in_channel;
                    int input_val =
                        input_data[Offset(input_shape, 0, out_y + filter_y, out_x + filter_x, in_channel)]; // Assumes batch size of 1
                    matrix_a[matrix_a_row][matrix_a_col] =
                        input_val + input_offset;
                }
            }
        }
    }
}

// Fill matrix_b with values from filter_data.
for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                int matrix_b_row = (filter_y * filter_width * input_depth) +
                                   (filter_x * input_depth) + in_channel;
                int matrix_b_col = out_channel;
                int filter_val = filter_data[Offset(filter_shape, out_channel,
                                                    filter_y, filter_x, in_channel)];
                matrix_b[matrix_b_row][matrix_b_col] = filter_val;
            }
        }
    }
}

// Perform matrix multiplication.
// for (int i = 0; i < matrix_a_rows; i++) {
//     for (int j = 0; j < matrix_b_cols; j++) {
//         int32_t acc = 0;
//         for (int k = 0; k < matrix_a_cols; k++) {
//             // acc += matrix_a[i][k] * static_cast<int32_t>(matrix_b[k][j]);
//             acc += matrix_a[i][k] * (matrix_b[k][j]);
//         }
//         output_matrix[i][j] = acc;
//     }
// }


// using cfu_op,2 send data,3 receive data
for(int i=0; i<matrix_a_rows ; i = i+4){
  for(int j=0 ; j<matrix_b_cols ; j = j+4){
    for(int k=0 ; k<matrix_a_cols ; k = k+ 1 ){
      uint8_t ainput0 = matrix_a[i][k];
      uint8_t ainput1 = (i+1<matrix_a_rows)? matrix_a[i+1][k] : 0;
      uint8_t ainput2 = (i+2<matrix_a_rows)? matrix_a[i+2][k] : 0;
      uint8_t ainput3 = (i+3<matrix_a_rows)? matrix_a[i+3][k] : 0;
      uint32_t ainput = (ainput0<<24)|(ainput1<<16)|(ainput2<<8)|(ainput3);

      uint8_t binput0 = matrix_b[k][j];
      uint8_t binput1 = (j+1<matrix_b_cols)? matrix_b[k][j+1] : 0;
      uint8_t binput2 = (j+2<matrix_b_cols)? matrix_b[k][j+2] : 0;
      uint8_t binput3 = (j+3<matrix_b_cols)? matrix_b[k][j+3] : 0;
      uint32_t binput = (binput0<<24)|(binput1<<16)|(binput2<<8)|(binput3);

      cfu_op2(0,ainput,binput);
    }
    for(int idx = 0; idx<4 ; idx++){
      for(int idy = 0; idy < 4; idy++){
        int32_t result_tmp = cfu_op3(0,idx,idy);
        output_matrix[i+idx][j+idy] = result_tmp;
        printf("%ld ",result_tmp);
        // printf("%ld ",output_matrix[i+idx][j+idy] );
      }
      printf("\n");
    }
    cfu_op4(0,0,0);//reset reg in cfu
  }
}

// Convert and quantize the output matrix to output_data.
for (int out_y = 0; out_y < output_height; ++out_y) {
    for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
            int matrix_c_row = out_y * output_width + out_x;
            int matrix_c_col = out_channel;
            int32_t acc = output_matrix[matrix_c_row][matrix_c_col];
            if (bias_data) {
                acc += bias_data[out_channel];
            }
            acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_channel], output_shift[out_channel]);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output_data[Offset(output_shape, 0, out_y, out_x, out_channel)] = static_cast<int8_t>(acc);
        }
    }
  }
  // printf("exit at : %u\n",perf_get_mcycle());
  perf_disable_counter(0);
} 

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter

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
  printf("2\n");
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

