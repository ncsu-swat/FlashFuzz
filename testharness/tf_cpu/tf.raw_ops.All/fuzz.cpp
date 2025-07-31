#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << "Error: " << message << std::endl;
    }
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
    case tensorflow::DT_BOOL:
      fillTensorWithData<bool>(tensor, data, offset, total_size);
      break;
    default:
      return;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t input_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        tensorflow::TensorShape input_tensor_shape;
        for (int64_t dim : input_shape) {
            input_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_tensor(tensorflow::DT_BOOL, input_tensor_shape);
        fillTensorWithDataByType(input_tensor, tensorflow::DT_BOOL, data, offset, size);
        
        if (offset >= size) return 0;
        
        uint8_t axis_rank = parseRank(data[offset++]);
        std::vector<int64_t> axis_shape = parseShape(data, offset, size, axis_rank);
        
        tensorflow::TensorShape axis_tensor_shape;
        for (int64_t dim : axis_shape) {
            axis_tensor_shape.AddDim(dim);
        }
        
        tensorflow::DataType axis_dtype = (offset < size && data[offset] % 2 == 0) ? 
            tensorflow::DT_INT32 : tensorflow::DT_INT64;
        offset++;
        
        tensorflow::Tensor axis_tensor(axis_dtype, axis_tensor_shape);
        
        if (axis_dtype == tensorflow::DT_INT32) {
            auto flat = axis_tensor.flat<int32_t>();
            for (int i = 0; i < flat.size(); ++i) {
                if (offset + sizeof(int32_t) <= size) {
                    int32_t value;
                    std::memcpy(&value, data + offset, sizeof(int32_t));
                    offset += sizeof(int32_t);
                    value = value % (2 * input_rank + 1) - input_rank;
                    flat(i) = value;
                } else {
                    flat(i) = 0;
                }
            }
        } else {
            auto flat = axis_tensor.flat<int64_t>();
            for (int i = 0; i < flat.size(); ++i) {
                if (offset + sizeof(int64_t) <= size) {
                    int64_t value;
                    std::memcpy(&value, data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    value = value % (2 * input_rank + 1) - input_rank;
                    flat(i) = value;
                } else {
                    flat(i) = 0;
                }
            }
        }
        
        bool keep_dims = (offset < size) ? (data[offset] % 2 == 1) : false;
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_BOOL);
        auto axis_placeholder = tensorflow::ops::Placeholder(root, axis_dtype);
        
        auto all_op = tensorflow::ops::All(root, input_placeholder, axis_placeholder, 
                                          tensorflow::ops::All::KeepDims(keep_dims));
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({{input_placeholder, input_tensor}, 
                                                 {axis_placeholder, axis_tensor}}, 
                                                {all_op}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}