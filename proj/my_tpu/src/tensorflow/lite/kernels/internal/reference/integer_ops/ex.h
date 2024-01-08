#include <vector>

// Helper function to perform im2col transformation.
void Im2Col(const int8_t* input_data, const int input_height, const int input_width,
            const int filter_height, const int filter_width,
            const int stride_height, const int stride_width,
            std::vector<int8_t>& im2col_data) {
  // Calculate the dimensions of the output matrix.
  int output_height = (input_height - filter_height) / stride_height + 1;
  int output_width = (input_width - filter_width) / stride_width + 1;
  
  // Initialize im2col_data with zeros.
  im2col_data.resize(filter_height * filter_width * output_height * output_width, 0);

  for (int i = 0; i < output_height; ++i) {
    for (int j = 0; j < output_width; ++j) {
      for (int y = 0; y < filter_height; ++y) {
        for (int x = 0; x < filter_width; ++x) {
          int im2col_idx = ((i * output_width) + j) *
                               (filter_height * filter_width) +
                           (y * filter_width + x);
          int input_idx = (i * stride_height + y) * input_width + (j * stride_width + x);
          im2col_data[im2col_idx] = input_data[input_idx];
        }
      }
    }
  }
}

// Perform convolution using im2col and matrix multiplication.
void ConvIm2ColMatrixMul(const ConvParams& params, const int32_t* output_multiplier,
                const int32_t* output_shift, const RuntimeShape& input_shape,
                const int8_t* input_data, const RuntimeShape& filter_shape,
                const int8_t* filter_data, const RuntimeShape& bias_shape,
                const int32_t* bias_data, const RuntimeShape& output_shape,
                int8_t* output_data) {
  // Perform im2col transformation on the input data.
  int input_height = input_shape.Dims(1);
  int input_width = input_shape.Dims(2);
  int filter_height = filter_shape.Dims(1);
  int filter_width = filter_shape.Dims(2);
  int stride_height = params.stride_height;
  int stride_width = params.stride_width;
  std::vector<int8_t> im2col_data;
  Im2Col(input_data, input_height, input_width, filter_height, filter_width,
         stride_height, stride_width, im2col_data);
  
  // Calculate the dimensions of the weight matrix.
  int output_channels = output_shape.Dims(3);
  int weight_rows = filter_height * filter_width;
  int weight_cols = output_channels;

  // Create matrices for im2col data and filter weights.
  std::vector<int32_t> im2col_matrix(im2col_data.begin(), im2col_data.end());
  std::vector<int32_t> filter_matrix(filter_data, filter_data + filter_shape.FlatSize());

  // Perform matrix multiplication.
  std::vector<int32_t> output_matrix(im2col_matrix.size() / weight_rows * weight_cols, 0);
  for (int i = 0; i < im2col_matrix.size() / weight_rows; ++i) {
    for (int j = 0; j < weight_cols; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < weight_rows; ++k) {
        acc += im2col_matrix[i * weight_rows + k] * filter_matrix[k * weight_cols + j];
      }
      output_matrix[i * weight_cols + j] = acc;
    }
  }
  
  // Add bias if available.
  if (bias_data) {
    for (int i = 0; i < output_channels; ++i) {
      int32_t bias = bias_data[i];
      for (int j = 0; j < im2col_matrix.size() / weight_rows; ++j) {
        output_matrix[j * weight_cols + i] += bias;
      }
    }
  }
  
  // Perform output quantization and write to output_data.
  for (int i = 0; i < im2col_matrix.size() / weight_rows; ++i) {
    for (int j = 0; j < weight_cols; ++j) {
      int32_t acc = output_matrix[i * weight_cols + j];
      acc = MultiplyByQuantizedMultiplier(
          acc, output_multiplier[j], output_shift[j]);
      acc += params.output_offset;
      acc = std::max(acc, params.quantized_activation_min);
      acc = std::min(acc, params.quantized_activation_max);
      output_data[i * weight_cols + j] = static_cast<int8_t>(acc);
    }
  }
}
