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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include <stdio.h>
#include "perf.h"
#include "cfu.h"

namespace tflite {
namespace reference_integer_ops {

// For per-channel functions, since it is defined in quantization spec that
// weights are symmetric
// (https://www.tensorflow.org/lite/performance/quantization_spec#symmetric_vs_asymmetric),
// zero_point (params.weights_offset) is always 0.
// However, for per-tensor functions, params.weights_offset is still applied for
// backward compatibility.

inline void FullyConnectedPerChannel(
    const FullyConnectedParams& params, const int32_t* output_multiplier,
    const int* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      int32_t acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += filter_val * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_c],
                                          output_shift[out_c]);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int8_t>(acc);
    }
  }
}

template <typename AccumScalar>
inline void FullyConnectedPerChannel(
    const FullyConnectedParams& params, const int32_t* output_multiplier,
    const int* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      AccumScalar acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += filter_val * input_val;
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled = MultiplyByQuantizedMultiplier(
          acc, output_multiplier[out_c], output_shift[out_c]);
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int16_t>(acc_scaled);
    }
  }
}

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  const int buffer_size = 200;
  const int iterations = (accum_depth%buffer_size==0) ? accum_depth/buffer_size : accum_depth/buffer_size + 1;
  int32_t res[5] = {0,0,0,0,0};
  uint32_t KM;
  uint32_t N;
  int32_t tmp0=0, tmp1=0, tmp2=0, tmp3=0;
  printf("Entering loop at: %u\n", perf_get_mcycle());
  perf_enable_counter(0);
  for(int i=0; i<iterations; ++i){
    for(int d=0; d < buffer_size && i*buffer_size+d < accum_depth; ++d){
      uint32_t input = (input_data[i*buffer_size + d] + input_offset)<<24;
      uint8_t filter0 = (filter_data[i*buffer_size + d] + filter_offset);
      uint8_t filter1 = (filter_data[accum_depth + i*buffer_size + d] + filter_offset);
      uint8_t filter2 = (filter_data[2 * accum_depth + i*buffer_size + d] + filter_offset);
      uint8_t filter3 = (filter_data[3 * accum_depth + i*buffer_size + d] + filter_offset);
      uint32_t filter = (filter0<<24)|(filter1<<16)|(filter2<<8)|(filter3);
      cfu_op1(0, input, filter);
      //if(i==0 && (uint8_t)(input>>24)!=0 && (int8_t)(filter)!=0)
      //  printf("%d,%d\n", (uint8_t)(input>>24), (int8_t)(filter0));
    }
    KM = ((buffer_size)<<16)|1;
    N = 4;
    cfu_op3(0, KM, N);
    res[4] += 1;
    tmp0 = cfu_op2(0, 0, 0);
    tmp0 = cfu_op2(0, 0, 0);
    res[0] += tmp0;
    tmp1 = cfu_op2(0, 0, 1);
    tmp1 = cfu_op2(0, 0, 1);
    res[1] += tmp1;
    tmp2 = cfu_op2(0, 0, 2);
    tmp2 = cfu_op2(0, 0, 2);
    res[2] += tmp2;
    tmp3 = cfu_op2(0, 0, 3);
    tmp3 = cfu_op2(0, 0, 3);
    res[3] += tmp3;
    //printf("%d %d %d %d\n\n\n", (int)tmp0, (int)tmp1, (int)tmp2, (int)tmp3);
    //printf("%d %d %d %d\n", (int)res[0], (int)res[1], (int)res[2], (int)res[3]);
  }
  
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      //int32_t acc = 0;
      //for (int d = 0; d < accum_depth; ++d) {
      //  int32_t input_val = input_data[b * accum_depth + d];
      //  int32_t filter_val = filter_data[out_c * accum_depth + d];
      //  acc += (filter_val + filter_offset) * (input_val + input_offset);
      //  if(out_c==0 && d>=3*buffer_size &&(input_val + input_offset)!=0 && (filter_val + filter_offset)!=0)
      //     printf("%d,%d\n", (int)(input_val + input_offset), (int)(filter_val + filter_offset));
      //}
      // printf("%d\n", (int)acc);
      if (bias_data) {
        //acc += bias_data[out_c];
	res[out_c] += bias_data[out_c];
      }
      //acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      //acc += output_offset;
      //acc = std::max(acc, output_activation_min);
      //acc = std::min(acc, output_activation_max);
      //output_data[out_c + output_depth * b] = static_cast<int8_t>(acc);
      res[out_c] = MultiplyByQuantizedMultiplier(res[out_c], output_multiplier, output_shift);
      res[out_c] += output_offset;
      res[out_c] = std::max(res[out_c], output_activation_min);
      res[out_c] = std::min(res[out_c], output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int8_t>(res[out_c]); 
    }
  }
  perf_disable_counter(0);
  printf("Exiting loop at: %u\n", perf_get_mcycle());
}

template <typename AccumScalar>
inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      AccumScalar acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * input_val;
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled =
          MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int16_t>(acc_scaled);
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_
