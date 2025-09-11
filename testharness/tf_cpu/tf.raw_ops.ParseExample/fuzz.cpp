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
#include <cstring>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10
#define MAX_LIST_SIZE 5

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
            uint8_t str_len = data[offset] % 20;
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_sparse_keys = (data[offset++] % MAX_LIST_SIZE) + 1;
        uint8_t num_dense_keys = (data[offset++] % MAX_LIST_SIZE) + 1;
        uint8_t batch_size = (data[offset++] % 5) + 1;

        tensorflow::TensorShape serialized_shape({batch_size});
        tensorflow::Tensor serialized_tensor(tensorflow::DT_STRING, serialized_shape);
        fillStringTensor(serialized_tensor, data, offset, size);

        tensorflow::TensorShape names_shape({batch_size});
        tensorflow::Tensor names_tensor(tensorflow::DT_STRING, names_shape);
        fillStringTensor(names_tensor, data, offset, size);

        std::vector<tensorflow::Input> sparse_keys;
        std::vector<tensorflow::DataType> sparse_types;
        for (uint8_t i = 0; i < num_sparse_keys; ++i) {
            tensorflow::Tensor sparse_key_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
            fillStringTensor(sparse_key_tensor, data, offset, size);
            sparse_keys.push_back(tensorflow::Input(sparse_key_tensor));
            
            tensorflow::DataType sparse_type = parseDataType(data[offset++]);
            sparse_types.push_back(sparse_type);
        }

        std::vector<tensorflow::Input> dense_keys;
        std::vector<tensorflow::Input> dense_defaults;
        std::vector<tensorflow::PartialTensorShape> dense_shapes;
        
        for (uint8_t i = 0; i < num_dense_keys; ++i) {
            tensorflow::Tensor dense_key_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
            fillStringTensor(dense_key_tensor, data, offset, size);
            dense_keys.push_back(tensorflow::Input(dense_key_tensor));

            tensorflow::DataType dense_type = parseDataType(data[offset++]);
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape default_shape(shape);
            tensorflow::Tensor default_tensor(dense_type, default_shape);
            fillTensorWithDataByType(default_tensor, dense_type, data, offset, size);
            dense_defaults.push_back(tensorflow::Input(default_tensor));
            
            dense_shapes.push_back(tensorflow::PartialTensorShape(shape));
        }

        auto parse_example = tensorflow::ops::ParseExample(
            root,
            tensorflow::Input(serialized_tensor),
            tensorflow::Input(names_tensor),
            tensorflow::InputList(sparse_keys),
            tensorflow::InputList(dense_keys),
            tensorflow::InputList(dense_defaults),
            sparse_types,
            dense_shapes
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        std::vector<tensorflow::Output> all_outputs;
        for (const auto& sparse_idx : parse_example.sparse_indices) {
            all_outputs.push_back(sparse_idx);
        }
        for (const auto& sparse_val : parse_example.sparse_values) {
            all_outputs.push_back(sparse_val);
        }
        for (const auto& sparse_shape : parse_example.sparse_shapes) {
            all_outputs.push_back(sparse_shape);
        }
        for (const auto& dense_val : parse_example.dense_values) {
            all_outputs.push_back(dense_val);
        }

        tensorflow::Status status = session.Run(all_outputs, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
