#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/sparse_ops.h"
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
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 11) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
            dtype = tensorflow::DT_INT32;
            break;
        case 3:
            dtype = tensorflow::DT_UINT8;
            break;
        case 4:
            dtype = tensorflow::DT_INT16;
            break;
        case 5:
            dtype = tensorflow::DT_INT8;
            break;
        case 6:
            dtype = tensorflow::DT_INT64;
            break;
        case 7:
            dtype = tensorflow::DT_BOOL;
            break;
        case 8:
            dtype = tensorflow::DT_UINT16;
            break;
        case 9:
            dtype = tensorflow::DT_UINT32;
            break;
        case 10:
            dtype = tensorflow::DT_UINT64;
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
        case tensorflow::DT_UINT8:
            fillTensorWithData<uint8_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT16:
            fillTensorWithData<int16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT8:
            fillTensorWithData<int8_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BOOL:
            fillTensorWithData<bool>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT16:
            fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT32:
            fillTensorWithData<uint32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT64:
            fillTensorWithData<uint64_t>(tensor, data, offset, total_size);
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
        tensorflow::DataType values_dtype = parseDataType(data[offset++]);
        
        if (offset >= size) return 0;
        uint8_t num_indices = data[offset++] % 10 + 1;
        
        if (offset >= size) return 0;
        uint8_t dense_shape_rank = data[offset++] % 3 + 1;
        
        std::vector<int64_t> dense_shape_dims = parseShape(data, offset, size, dense_shape_rank);
        if (dense_shape_dims.empty()) {
            dense_shape_dims.push_back(5);
        }
        
        int64_t total_rows = dense_shape_dims[0];
        if (total_rows <= 0) total_rows = 5;
        
        tensorflow::TensorShape indices_shape({num_indices, dense_shape_rank});
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT64, indices_shape);
        fillTensorWithData<int64_t>(indices_tensor, data, offset, size);
        
        auto indices_flat = indices_tensor.flat<int64_t>();
        for (int i = 0; i < indices_flat.size(); ++i) {
            int64_t dim_idx = i % dense_shape_rank;
            int64_t max_val = dense_shape_dims[dim_idx] - 1;
            if (max_val < 0) max_val = 0;
            indices_flat(i) = std::abs(indices_flat(i)) % (max_val + 1);
        }
        
        tensorflow::TensorShape values_shape({num_indices});
        tensorflow::Tensor values_tensor(values_dtype, values_shape);
        fillTensorWithDataByType(values_tensor, values_dtype, data, offset, size);
        
        tensorflow::TensorShape dense_shape_tensor_shape({dense_shape_rank});
        tensorflow::Tensor dense_shape_tensor(tensorflow::DT_INT64, dense_shape_tensor_shape);
        auto dense_shape_flat = dense_shape_tensor.flat<int64_t>();
        for (int i = 0; i < dense_shape_rank; ++i) {
            dense_shape_flat(i) = dense_shape_dims[i];
        }
        
        tensorflow::TensorShape default_value_shape({});
        tensorflow::Tensor default_value_tensor(values_dtype, default_value_shape);
        fillTensorWithDataByType(default_value_tensor, values_dtype, data, offset, size);
        
        auto indices_input = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        auto values_input = tensorflow::ops::Placeholder(root, values_dtype);
        auto dense_shape_input = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        auto default_value_input = tensorflow::ops::Placeholder(root, values_dtype);
        
        auto sparse_fill_empty_rows = tensorflow::ops::SparseFillEmptyRows(
            root, indices_input, values_input, dense_shape_input, default_value_input);
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({
            {indices_input, indices_tensor},
            {values_input, values_tensor},
            {dense_shape_input, dense_shape_tensor},
            {default_value_input, default_value_tensor}
        }, {
            sparse_fill_empty_rows.output_indices,
            sparse_fill_empty_rows.output_values,
            sparse_fill_empty_rows.empty_row_indicator,
            sparse_fill_empty_rows.reverse_index_map
        }, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
