#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/sparse_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include <iostream>
#include <vector>
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
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_INT64;
            break;
        case 1:
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
                char c = 'a' + (data[offset] % 26);
                str += c;
                offset++;
            }
            flat(i) = tensorflow::tstring(str);
        } else {
            flat(i) = tensorflow::tstring("a");
        }
    }
}

void fillTensorWithDataByType(tensorflow::Tensor& tensor,
                              tensorflow::DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
    switch (dtype) {
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
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_sparse = (data[offset] % 3) + 1;
        offset++;
        
        uint8_t num_dense = (data[offset] % 3) + 1;
        offset++;

        std::vector<tensorflow::Output> indices_list;
        std::vector<tensorflow::Output> values_list;
        std::vector<tensorflow::Output> shapes_list;
        std::vector<tensorflow::Output> dense_inputs_list;

        for (uint8_t i = 0; i < num_sparse; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType values_dtype = parseDataType(data[offset]);
            offset++;
            
            uint8_t indices_rank = 2;
            std::vector<int64_t> indices_shape = {2, 2};
            tensorflow::Tensor indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(indices_shape));
            fillTensorWithData<int64_t>(indices_tensor, data, offset, size);
            
            auto indices_const = tensorflow::ops::Const(root, indices_tensor);
            indices_list.push_back(indices_const);

            std::vector<int64_t> values_shape = {2};
            tensorflow::Tensor values_tensor(values_dtype, tensorflow::TensorShape(values_shape));
            fillTensorWithDataByType(values_tensor, values_dtype, data, offset, size);
            
            auto values_const = tensorflow::ops::Const(root, values_tensor);
            values_list.push_back(values_const);

            std::vector<int64_t> shape_shape = {2};
            tensorflow::Tensor shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(shape_shape));
            auto shape_flat = shape_tensor.flat<int64_t>();
            shape_flat(0) = 2;
            shape_flat(1) = 2;
            
            auto shape_const = tensorflow::ops::Const(root, shape_tensor);
            shapes_list.push_back(shape_const);
        }

        for (uint8_t i = 0; i < num_dense; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType dense_dtype = parseDataType(data[offset]);
            offset++;
            
            std::vector<int64_t> dense_shape = {2, 1};
            tensorflow::Tensor dense_tensor(dense_dtype, tensorflow::TensorShape(dense_shape));
            fillTensorWithDataByType(dense_tensor, dense_dtype, data, offset, size);
            
            auto dense_const = tensorflow::ops::Const(root, dense_tensor);
            dense_inputs_list.push_back(dense_const);
        }

        tensorflow::Tensor sep_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        auto sep_scalar = sep_tensor.scalar<tensorflow::tstring>();
        sep_scalar() = tensorflow::tstring("_X_");
        auto sep_const = tensorflow::ops::Const(root, sep_tensor);

        if (indices_list.empty() || values_list.empty() || shapes_list.empty()) {
            return 0;
        }

        auto sparse_cross = tensorflow::ops::SparseCrossV2(
            root,
            indices_list,
            values_list,
            shapes_list,
            dense_inputs_list,
            sep_const
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {sparse_cross.output_indices, sparse_cross.output_values, sparse_cross.output_shape},
            &outputs
        );
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}