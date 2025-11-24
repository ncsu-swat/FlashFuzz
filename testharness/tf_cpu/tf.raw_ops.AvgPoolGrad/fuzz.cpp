#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseGradDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 2:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 3:
            dtype = tensorflow::DT_DOUBLE;
            break;
    }
    return dtype;
}

uint8_t parseRank(uint8_t byte) {
    constexpr uint8_t range = MAX_RANK - MIN_RANK + 1;
    uint8_t rank = byte % range + MIN_RANK;
    return rank;
}

std::vector<int64_t> parseShape(const uint8_t* data, size_t& offset, size_t total_size, uint8_t rank) {
    if (rank == 0) {
        return {};
    }

    std::vector<int64_t> shape;
    shape.reserve(rank);
    const auto sizeof_dim = sizeof(int64_t);

    for (uint8_t i = 0; i < rank; ++i) {
        if (offset + sizeof_dim <= total_size) {
            int64_t dim_val;
            std::memcpy(&dim_val, data + offset, sizeof_dim);
            offset += sizeof_dim;
            
            dim_val = MIN_TENSOR_SHAPE_DIMS_TF +
                    static_cast<int64_t>((static_cast<uint64_t>(std::abs(dim_val)) %
                                        static_cast<uint64_t>(MAX_TENSOR_SHAPE_DIMS_TF - MIN_TENSOR_SHAPE_DIMS_TF + 1)));

            shape.push_back(dim_val);
        } else {
             shape.push_back(1);
        }
    }

    return shape;
}

template <typename T>
void fillTensorWithData(tensorflow::Tensor& tensor, const uint8_t* data,
                        size_t& offset, size_t total_size) {
  auto flat = tensor.flat<T>();
  const size_t num_elements = flat.size();
  const size_t element_size = sizeof(T);

  for (size_t i = 0; i < num_elements; ++i) {
    if (offset + element_size <= total_size) {
      T value;
      std::memcpy(&value, data + offset, element_size);
      offset += element_size;
      flat(i) = value;
    } else {
      flat(i) = T{};
    }
  }
}

void fillTensorWithDataByType(tensorflow::Tensor& tensor,
                              tensorflow::DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
  switch (dtype) {
    case tensorflow::DT_FLOAT:
      fillTensorWithData<float>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_DOUBLE:
      fillTensorWithData<double>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT32:
      fillTensorWithData<int32_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_BFLOAT16:
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset,
                                               total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType grad_dtype = parseGradDataType(data[offset++]);
        
        std::vector<int64_t> orig_input_shape_data;
        if (offset + 4 * sizeof(int32_t) <= size) {
            for (int i = 0; i < 4; ++i) {
                int32_t dim;
                std::memcpy(&dim, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                dim = std::abs(dim) % 10 + 1;
                orig_input_shape_data.push_back(static_cast<int64_t>(dim));
            }
        } else {
            orig_input_shape_data = {1, 4, 4, 1};
        }
        
        tensorflow::Tensor orig_input_shape_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({4}));
        auto orig_input_flat = orig_input_shape_tensor.flat<int32_t>();
        for (int i = 0; i < 4; ++i) {
            orig_input_flat(i) = static_cast<int32_t>(orig_input_shape_data[i]);
        }
        
        std::vector<int> ksize;
        if (offset + 4 * sizeof(int32_t) <= size) {
            for (int i = 0; i < 4; ++i) {
                int32_t k;
                std::memcpy(&k, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                k = std::abs(k) % 5 + 1;
                ksize.push_back(k);
            }
        } else {
            ksize = {1, 2, 2, 1};
        }
        
        std::vector<int> strides;
        if (offset + 4 * sizeof(int32_t) <= size) {
            for (int i = 0; i < 4; ++i) {
                int32_t s;
                std::memcpy(&s, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                s = std::abs(s) % 3 + 1;
                strides.push_back(s);
            }
        } else {
            strides = {1, 1, 1, 1};
        }
        
        std::string padding = (offset < size && data[offset++] % 2 == 0) ? "SAME" : "VALID";
        std::string data_format = (offset < size && data[offset++] % 2 == 0) ? "NHWC" : "NCHW";
        
        auto orig_input_shape_op = tensorflow::ops::Const(root, orig_input_shape_tensor);

        // Compute an output shape compatible with the AvgPool forward op so the
        // gradient tensor lines up with the expected output dimensions.
        std::vector<int64_t> out_shape = {orig_input_shape_data[0], 1, 1, 1};
        if (orig_input_shape_data.size() == 4) {
            if (data_format == "NHWC") {
                int64_t batch = orig_input_shape_data[0];
                int64_t height = orig_input_shape_data[1];
                int64_t width = orig_input_shape_data[2];
                int64_t channels = orig_input_shape_data[3];

                int64_t out_height, out_width;
                if (padding == "VALID") {
                    out_height = (height - ksize[1]) / strides[1] + 1;
                    out_width = (width - ksize[2]) / strides[2] + 1;
                } else {
                    out_height = (height + strides[1] - 1) / strides[1];
                    out_width = (width + strides[2] - 1) / strides[2];
                }

                if (out_height > 0 && out_width > 0) {
                    out_shape = {batch, out_height, out_width, channels};
                }
            } else {
                int64_t batch = orig_input_shape_data[0];
                int64_t channels = orig_input_shape_data[1];
                int64_t height = orig_input_shape_data[2];
                int64_t width = orig_input_shape_data[3];

                int64_t out_height, out_width;
                if (padding == "VALID") {
                    out_height = (height - ksize[2]) / strides[2] + 1;
                    out_width = (width - ksize[3]) / strides[3] + 1;
                } else {
                    out_height = (height + strides[2] - 1) / strides[2];
                    out_width = (width + strides[3] - 1) / strides[3];
                }

                if (out_height > 0 && out_width > 0) {
                    out_shape = {batch, channels, out_height, out_width};
                }
            }
        }

        tensorflow::TensorShape grad_tensor_shape(out_shape);
        tensorflow::Tensor grad_tensor_adjusted(grad_dtype, grad_tensor_shape);
        fillTensorWithDataByType(grad_tensor_adjusted, grad_dtype, data, offset, size);
        auto adjusted_grad_op = tensorflow::ops::Const(root, grad_tensor_adjusted);

        auto avg_pool_grad = tensorflow::ops::internal::AvgPoolGrad(
            root.WithOpName("AvgPoolGrad"),
            orig_input_shape_op,
            adjusted_grad_op,
            ksize,
            strides,
            padding,
            tensorflow::ops::internal::AvgPoolGrad::Attrs().DataFormat(data_format));

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({avg_pool_grad}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
