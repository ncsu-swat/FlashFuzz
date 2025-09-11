#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/io_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
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
            dtype = tensorflow::DT_FLOAT;
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
            uint8_t str_len = data[offset] % 20 + 1;
            offset++;
            
            std::string str;
            for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                str += static_cast<char>(data[offset] % 128);
                offset++;
            }
            flat(i) = tensorflow::tstring(str);
        } else {
            flat(i) = tensorflow::tstring("default");
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
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType ckpt_path_dtype = tensorflow::DT_STRING;
        uint8_t ckpt_path_rank = 0;
        std::vector<int64_t> ckpt_path_shape = {};
        tensorflow::TensorShape ckpt_path_tensor_shape(ckpt_path_shape);
        tensorflow::Tensor ckpt_path_tensor(ckpt_path_dtype, ckpt_path_tensor_shape);
        fillTensorWithDataByType(ckpt_path_tensor, ckpt_path_dtype, data, offset, size);

        tensorflow::DataType old_tensor_name_dtype = tensorflow::DT_STRING;
        uint8_t old_tensor_name_rank = 0;
        std::vector<int64_t> old_tensor_name_shape = {};
        tensorflow::TensorShape old_tensor_name_tensor_shape(old_tensor_name_shape);
        tensorflow::Tensor old_tensor_name_tensor(old_tensor_name_dtype, old_tensor_name_tensor_shape);
        fillTensorWithDataByType(old_tensor_name_tensor, old_tensor_name_dtype, data, offset, size);

        if (offset >= size) return 0;
        uint8_t row_remapping_rank = parseRank(data[offset++]);
        std::vector<int64_t> row_remapping_shape = parseShape(data, offset, size, row_remapping_rank);
        tensorflow::TensorShape row_remapping_tensor_shape(row_remapping_shape);
        tensorflow::Tensor row_remapping_tensor(tensorflow::DT_INT64, row_remapping_tensor_shape);
        fillTensorWithDataByType(row_remapping_tensor, tensorflow::DT_INT64, data, offset, size);

        if (offset >= size) return 0;
        uint8_t col_remapping_rank = parseRank(data[offset++]);
        std::vector<int64_t> col_remapping_shape = parseShape(data, offset, size, col_remapping_rank);
        tensorflow::TensorShape col_remapping_tensor_shape(col_remapping_shape);
        tensorflow::Tensor col_remapping_tensor(tensorflow::DT_INT64, col_remapping_tensor_shape);
        fillTensorWithDataByType(col_remapping_tensor, tensorflow::DT_INT64, data, offset, size);

        if (offset >= size) return 0;
        uint8_t initializing_values_rank = parseRank(data[offset++]);
        std::vector<int64_t> initializing_values_shape = parseShape(data, offset, size, initializing_values_rank);
        tensorflow::TensorShape initializing_values_tensor_shape(initializing_values_shape);
        tensorflow::Tensor initializing_values_tensor(tensorflow::DT_FLOAT, initializing_values_tensor_shape);
        fillTensorWithDataByType(initializing_values_tensor, tensorflow::DT_FLOAT, data, offset, size);

        if (offset + sizeof(int64_t) * 3 > size) return 0;
        int64_t num_rows, num_cols, max_rows_in_memory;
        std::memcpy(&num_rows, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        std::memcpy(&num_cols, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        std::memcpy(&max_rows_in_memory, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);

        num_rows = std::abs(num_rows) % 10 + 1;
        num_cols = std::abs(num_cols) % 10 + 1;
        max_rows_in_memory = std::abs(max_rows_in_memory) % 10 - 1;

        auto ckpt_path_placeholder = tensorflow::ops::Placeholder(root, ckpt_path_dtype);
        auto old_tensor_name_placeholder = tensorflow::ops::Placeholder(root, old_tensor_name_dtype);
        auto row_remapping_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        auto col_remapping_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        auto initializing_values_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);

        // Create the operation using raw_ops
        tensorflow::Output load_and_remap_matrix = tensorflow::Operation(
            root.WithOpName("LoadAndRemapMatrix"),
            "LoadAndRemapMatrix",
            {ckpt_path_placeholder, old_tensor_name_placeholder, row_remapping_placeholder, 
             col_remapping_placeholder, initializing_values_placeholder},
            {tensorflow::DataType::DT_FLOAT},
            {{"num_rows", num_rows}, {"num_cols", num_cols}, {"max_rows_in_memory", max_rows_in_memory}}
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;

        tensorflow::Status status = session.Run({
            {ckpt_path_placeholder, ckpt_path_tensor},
            {old_tensor_name_placeholder, old_tensor_name_tensor},
            {row_remapping_placeholder, row_remapping_tensor},
            {col_remapping_placeholder, col_remapping_tensor},
            {initializing_values_placeholder, initializing_values_tensor}
        }, {load_and_remap_matrix}, &outputs);

        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
