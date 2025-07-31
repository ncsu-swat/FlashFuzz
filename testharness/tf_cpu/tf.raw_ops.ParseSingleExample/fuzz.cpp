#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/parsing_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
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

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
            break;
        case 2:
            dtype = tensorflow::DT_STRING;
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

void fillStringTensor(tensorflow::Tensor& tensor, const uint8_t* data,
                      size_t& offset, size_t total_size) {
    auto flat = tensor.flat<tensorflow::tstring>();
    const size_t num_elements = flat.size();

    for (size_t i = 0; i < num_elements; ++i) {
        if (offset < total_size) {
            uint8_t str_len = data[offset] % 10 + 1;
            offset++;
            
            std::string str;
            for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                str += static_cast<char>(data[offset]);
                offset++;
            }
            flat(i) = tensorflow::tstring(str);
        } else {
            flat(i) = tensorflow::tstring("");
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
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_STRING:
            fillStringTensor(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t serialized_len = data[offset] % 50 + 1;
        offset++;
        
        std::string serialized_str;
        for (uint8_t i = 0; i < serialized_len && offset < size; ++i) {
            serialized_str += static_cast<char>(data[offset]);
            offset++;
        }
        
        tensorflow::Tensor serialized_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        serialized_tensor.scalar<tensorflow::tstring>()() = tensorflow::tstring(serialized_str);

        uint8_t num_dense = data[offset] % 3 + 1;
        offset++;
        
        uint8_t num_sparse = data[offset] % 3;
        offset++;

        std::vector<tensorflow::Output> dense_defaults;
        std::vector<std::string> dense_keys;
        std::vector<tensorflow::PartialTensorShape> dense_shapes;
        
        for (uint8_t i = 0; i < num_dense && offset < size; ++i) {
            tensorflow::DataType dtype = parseDataType(data[offset]);
            offset++;
            
            uint8_t rank = parseRank(data[offset]);
            offset++;
            
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor default_tensor(dtype, tensor_shape);
            fillTensorWithDataByType(default_tensor, dtype, data, offset, size);
            
            auto default_const = tensorflow::ops::Const(root, default_tensor);
            dense_defaults.push_back(default_const);
            dense_keys.push_back("dense_key_" + std::to_string(i));
            dense_shapes.push_back(tensorflow::PartialTensorShape(shape));
        }

        std::vector<std::string> sparse_keys;
        std::vector<tensorflow::DataType> sparse_types;
        
        for (uint8_t i = 0; i < num_sparse && offset < size; ++i) {
            tensorflow::DataType dtype = parseDataType(data[offset]);
            offset++;
            
            sparse_keys.push_back("sparse_key_" + std::to_string(i));
            sparse_types.push_back(dtype);
        }

        auto parse_op = tensorflow::ops::ParseSingleExample(
            root,
            serialized_tensor,
            dense_defaults,
            num_sparse,
            sparse_keys,
            dense_keys,
            sparse_types,
            dense_shapes
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        std::vector<tensorflow::Output> all_outputs;
        for (const auto& idx : parse_op.sparse_indices) {
            all_outputs.push_back(idx);
        }
        for (const auto& val : parse_op.sparse_values) {
            all_outputs.push_back(val);
        }
        for (const auto& shape : parse_op.sparse_shapes) {
            all_outputs.push_back(shape);
        }
        for (const auto& dense : parse_op.dense_values) {
            all_outputs.push_back(dense);
        }

        tensorflow::Status status = session.Run(all_outputs, &outputs);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}