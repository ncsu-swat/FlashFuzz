#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/random_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseFloatDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 2:
            dtype = tensorflow::DT_DOUBLE;
            break;
    }
    return dtype;
}

tensorflow::DataType parseIntDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
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
    case tensorflow::DT_INT64:
      fillTensorWithData<int64_t>(tensor, data, offset, total_size);
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
        tensorflow::DataType shape_dtype = parseIntDataType(data[offset++]);
        tensorflow::DataType seed_dtype = parseIntDataType(data[offset++]);
        tensorflow::DataType float_dtype = parseFloatDataType(data[offset++]);
        
        uint8_t shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> shape_dims = parseShape(data, offset, size, shape_rank);
        
        tensorflow::TensorShape shape_tensor_shape({static_cast<int64_t>(shape_dims.size())});
        tensorflow::Tensor shape_tensor(shape_dtype, shape_tensor_shape);
        
        if (shape_dtype == tensorflow::DT_INT32) {
            auto flat = shape_tensor.flat<int32_t>();
            for (size_t i = 0; i < shape_dims.size(); ++i) {
                flat(i) = static_cast<int32_t>(shape_dims[i]);
            }
        } else {
            auto flat = shape_tensor.flat<int64_t>();
            for (size_t i = 0; i < shape_dims.size(); ++i) {
                flat(i) = shape_dims[i];
            }
        }
        
        tensorflow::TensorShape seed_shape({2});
        tensorflow::Tensor seed_tensor(seed_dtype, seed_shape);
        fillTensorWithDataByType(seed_tensor, seed_dtype, data, offset, size);
        
        int64_t total_elements = 1;
        for (auto dim : shape_dims) {
            total_elements *= dim;
        }
        
        tensorflow::TensorShape param_shape({});
        if (total_elements > 1) {
            param_shape = tensorflow::TensorShape({total_elements});
        }
        
        tensorflow::Tensor means_tensor(float_dtype, param_shape);
        tensorflow::Tensor stddevs_tensor(float_dtype, param_shape);
        tensorflow::Tensor minvals_tensor(float_dtype, param_shape);
        tensorflow::Tensor maxvals_tensor(float_dtype, param_shape);
        
        fillTensorWithDataByType(means_tensor, float_dtype, data, offset, size);
        fillTensorWithDataByType(stddevs_tensor, float_dtype, data, offset, size);
        fillTensorWithDataByType(minvals_tensor, float_dtype, data, offset, size);
        fillTensorWithDataByType(maxvals_tensor, float_dtype, data, offset, size);
        
        if (float_dtype == tensorflow::DT_FLOAT) {
            auto stddevs_flat = stddevs_tensor.flat<float>();
            auto minvals_flat = minvals_tensor.flat<float>();
            auto maxvals_flat = maxvals_tensor.flat<float>();
            for (int i = 0; i < stddevs_flat.size(); ++i) {
                stddevs_flat(i) = std::abs(stddevs_flat(i)) + 0.1f;
                if (minvals_flat(i) > maxvals_flat(i)) {
                    std::swap(minvals_flat(i), maxvals_flat(i));
                }
                maxvals_flat(i) = minvals_flat(i) + std::abs(maxvals_flat(i) - minvals_flat(i)) + 0.1f;
            }
        } else if (float_dtype == tensorflow::DT_DOUBLE) {
            auto stddevs_flat = stddevs_tensor.flat<double>();
            auto minvals_flat = minvals_tensor.flat<double>();
            auto maxvals_flat = maxvals_tensor.flat<double>();
            for (int i = 0; i < stddevs_flat.size(); ++i) {
                stddevs_flat(i) = std::abs(stddevs_flat(i)) + 0.1;
                if (minvals_flat(i) > maxvals_flat(i)) {
                    std::swap(minvals_flat(i), maxvals_flat(i));
                }
                maxvals_flat(i) = minvals_flat(i) + std::abs(maxvals_flat(i) - minvals_flat(i)) + 0.1;
            }
        } else if (float_dtype == tensorflow::DT_HALF) {
            auto stddevs_flat = stddevs_tensor.flat<Eigen::half>();
            auto minvals_flat = minvals_tensor.flat<Eigen::half>();
            auto maxvals_flat = maxvals_tensor.flat<Eigen::half>();
            for (int i = 0; i < stddevs_flat.size(); ++i) {
                float stddev_val = static_cast<float>(stddevs_flat(i));
                float minval_val = static_cast<float>(minvals_flat(i));
                float maxval_val = static_cast<float>(maxvals_flat(i));
                
                stddev_val = std::abs(stddev_val) + 0.1f;
                if (minval_val > maxval_val) {
                    std::swap(minval_val, maxval_val);
                }
                maxval_val = minval_val + std::abs(maxval_val - minval_val) + 0.1f;
                
                stddevs_flat(i) = Eigen::half(stddev_val);
                minvals_flat(i) = Eigen::half(minval_val);
                maxvals_flat(i) = Eigen::half(maxval_val);
            }
        }
        
        auto shape_input = tensorflow::ops::Const(root, shape_tensor);
        auto seed_input = tensorflow::ops::Const(root, seed_tensor);
        auto means_input = tensorflow::ops::Const(root, means_tensor);
        auto stddevs_input = tensorflow::ops::Const(root, stddevs_tensor);
        auto minvals_input = tensorflow::ops::Const(root, minvals_tensor);
        auto maxvals_input = tensorflow::ops::Const(root, maxvals_tensor);
        
        // Use raw operation instead of ops namespace
        auto result = tensorflow::ops::StatelessTruncatedNormal(
            root.WithOpName("StatelessParameterizedTruncatedNormal"),
            shape_input, seed_input,
            tensorflow::ops::StatelessTruncatedNormal::Attrs()
                .Dtype(float_dtype));
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({result}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
