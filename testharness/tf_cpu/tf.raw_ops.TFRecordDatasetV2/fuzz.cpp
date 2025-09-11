#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/dataset_ops.h"
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

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_STRING;
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
            uint8_t str_len = data[offset] % 32;
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
        case tensorflow::DT_STRING:
            fillStringTensor(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t filenames_rank = parseRank(data[offset++]);
        std::vector<int64_t> filenames_shape = parseShape(data, offset, size, filenames_rank);
        tensorflow::TensorShape filenames_tensor_shape(filenames_shape);
        tensorflow::Tensor filenames_tensor(tensorflow::DT_STRING, filenames_tensor_shape);
        fillTensorWithDataByType(filenames_tensor, tensorflow::DT_STRING, data, offset, size);

        uint8_t compression_rank = parseRank(data[offset++]);
        std::vector<int64_t> compression_shape = parseShape(data, offset, size, compression_rank);
        tensorflow::TensorShape compression_tensor_shape(compression_shape);
        tensorflow::Tensor compression_tensor(tensorflow::DT_STRING, compression_tensor_shape);
        fillTensorWithDataByType(compression_tensor, tensorflow::DT_STRING, data, offset, size);

        uint8_t buffer_rank = parseRank(data[offset++]);
        std::vector<int64_t> buffer_shape = parseShape(data, offset, size, buffer_rank);
        tensorflow::TensorShape buffer_tensor_shape(buffer_shape);
        tensorflow::Tensor buffer_tensor(tensorflow::DT_INT64, buffer_tensor_shape);
        fillTensorWithDataByType(buffer_tensor, tensorflow::DT_INT64, data, offset, size);

        uint8_t offsets_rank = parseRank(data[offset++]);
        std::vector<int64_t> offsets_shape = parseShape(data, offset, size, offsets_rank);
        tensorflow::TensorShape offsets_tensor_shape(offsets_shape);
        tensorflow::Tensor offsets_tensor(tensorflow::DT_INT64, offsets_tensor_shape);
        fillTensorWithDataByType(offsets_tensor, tensorflow::DT_INT64, data, offset, size);

        auto filenames_op = tensorflow::ops::Const(root, filenames_tensor);
        auto compression_op = tensorflow::ops::Const(root, compression_tensor);
        auto buffer_op = tensorflow::ops::Const(root, buffer_tensor);
        auto offsets_op = tensorflow::ops::Const(root, offsets_tensor);

        // Create the TFRecordDatasetV2 operation using the raw API
        tensorflow::NodeDef node_def;
        node_def.set_name("TFRecordDatasetV2");
        node_def.set_op("TFRecordDatasetV2");
        
        tensorflow::Status status;
        auto dataset_op = root.AddNode(node_def, &status);
        
        // Add inputs to the operation
        dataset_op = tensorflow::ops::WithInput(dataset_op, filenames_op);
        dataset_op = tensorflow::ops::WithInput(dataset_op, compression_op);
        dataset_op = tensorflow::ops::WithInput(dataset_op, buffer_op);
        dataset_op = tensorflow::ops::WithInput(dataset_op, offsets_op);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({dataset_op}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
