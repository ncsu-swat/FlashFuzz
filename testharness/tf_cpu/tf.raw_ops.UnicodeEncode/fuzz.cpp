#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/string_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <string>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseInputValuesDataType(uint8_t selector) {
    return tensorflow::DT_INT32;
}

tensorflow::DataType parseInputSplitsDataType(uint8_t selector) {
    switch (selector % 2) {
        case 0:
            return tensorflow::DT_INT32;
        case 1:
            return tensorflow::DT_INT64;
        default:
            return tensorflow::DT_INT32;
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
    case tensorflow::DT_INT32:
      fillTensorWithData<int32_t>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_INT64:
      fillTensorWithData<int64_t>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

std::string parseOutputEncoding(uint8_t selector) {
    switch (selector % 3) {
        case 0:
            return "UTF-8";
        case 1:
            return "UTF-16-BE";
        case 2:
            return "UTF-32-BE";
        default:
            return "UTF-8";
    }
}

std::string parseErrors(uint8_t selector) {
    switch (selector % 3) {
        case 0:
            return "ignore";
        case 1:
            return "replace";
        case 2:
            return "strict";
        default:
            return "replace";
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) {
        return 0;
    }

    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t input_values_rank = parseRank(data[offset++]);
        if (input_values_rank != 1) {
            input_values_rank = 1;
        }
        
        tensorflow::DataType input_values_dtype = parseInputValuesDataType(data[offset++]);
        std::vector<int64_t> input_values_shape = parseShape(data, offset, size, input_values_rank);
        
        tensorflow::TensorShape input_values_tensor_shape;
        for (int64_t dim : input_values_shape) {
            input_values_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_values_tensor(input_values_dtype, input_values_tensor_shape);
        fillTensorWithDataByType(input_values_tensor, input_values_dtype, data, offset, size);

        uint8_t input_splits_rank = parseRank(data[offset++]);
        if (input_splits_rank != 1) {
            input_splits_rank = 1;
        }
        
        tensorflow::DataType input_splits_dtype = parseInputSplitsDataType(data[offset++]);
        std::vector<int64_t> input_splits_shape = parseShape(data, offset, size, input_splits_rank);
        
        tensorflow::TensorShape input_splits_tensor_shape;
        for (int64_t dim : input_splits_shape) {
            input_splits_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_splits_tensor(input_splits_dtype, input_splits_tensor_shape);
        fillTensorWithDataByType(input_splits_tensor, input_splits_dtype, data, offset, size);

        std::string output_encoding = parseOutputEncoding(data[offset++]);
        std::string errors = parseErrors(data[offset++]);
        
        int replacement_char = 65533;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&replacement_char, data + offset, sizeof(int));
            offset += sizeof(int);
            replacement_char = std::abs(replacement_char) % 1114112;
        }

        auto input_values_op = tensorflow::ops::Const(root, input_values_tensor);
        auto input_splits_op = tensorflow::ops::Const(root, input_splits_tensor);

        // Use raw_ops namespace to access UnicodeEncode
        auto unicode_encode_op = tensorflow::ops::internal::UnicodeEncode(
            root,
            input_values_op,
            input_splits_op,
            output_encoding,
            errors,
            replacement_char
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({unicode_encode_op}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
