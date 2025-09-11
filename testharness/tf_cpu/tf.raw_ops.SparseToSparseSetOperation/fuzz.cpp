#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/sparse_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 6) {
        case 0:
            dtype = tensorflow::DT_INT8;
            break;
        case 1:
            dtype = tensorflow::DT_INT16;
            break;
        case 2:
            dtype = tensorflow::DT_INT32;
            break;
        case 3:
            dtype = tensorflow::DT_INT64;
            break;
        case 4:
            dtype = tensorflow::DT_UINT8;
            break;
        case 5:
            dtype = tensorflow::DT_UINT16;
            break;
        default:
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

void fillTensorWithDataByType(tensorflow::Tensor& tensor,
                              tensorflow::DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
    switch (dtype) {
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
        case tensorflow::DT_UINT16:
            fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_STRING:
            {
                auto flat = tensor.flat<tensorflow::tstring>();
                const size_t num_elements = flat.size();
                for (size_t i = 0; i < num_elements; ++i) {
                    if (offset < total_size) {
                        uint8_t str_len = data[offset] % 10 + 1;
                        offset++;
                        std::string str;
                        for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                            str += static_cast<char>(data[offset] % 26 + 'a');
                            offset++;
                        }
                        flat(i) = tensorflow::tstring(str);
                    } else {
                        flat(i) = tensorflow::tstring("a");
                    }
                }
            }
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
        tensorflow::DataType values_dtype = parseDataType(data[offset++]);
        
        uint8_t set1_indices_rank = 2;
        uint8_t set1_values_rank = 1;
        uint8_t set1_shape_rank = 1;
        uint8_t set2_indices_rank = 2;
        uint8_t set2_values_rank = 1;
        uint8_t set2_shape_rank = 1;

        std::vector<int64_t> set1_indices_shape = {3, 2};
        std::vector<int64_t> set1_values_shape = {3};
        std::vector<int64_t> set1_shape_shape = {2};
        std::vector<int64_t> set2_indices_shape = {2, 2};
        std::vector<int64_t> set2_values_shape = {2};
        std::vector<int64_t> set2_shape_shape = {2};

        tensorflow::Tensor set1_indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(set1_indices_shape));
        tensorflow::Tensor set1_values_tensor(values_dtype, tensorflow::TensorShape(set1_values_shape));
        tensorflow::Tensor set1_shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(set1_shape_shape));
        tensorflow::Tensor set2_indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(set2_indices_shape));
        tensorflow::Tensor set2_values_tensor(values_dtype, tensorflow::TensorShape(set2_values_shape));
        tensorflow::Tensor set2_shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(set2_shape_shape));

        fillTensorWithData<int64_t>(set1_indices_tensor, data, offset, size);
        fillTensorWithDataByType(set1_values_tensor, values_dtype, data, offset, size);
        fillTensorWithData<int64_t>(set1_shape_tensor, data, offset, size);
        fillTensorWithData<int64_t>(set2_indices_tensor, data, offset, size);
        fillTensorWithDataByType(set2_values_tensor, values_dtype, data, offset, size);
        fillTensorWithData<int64_t>(set2_shape_tensor, data, offset, size);

        auto set1_indices_flat = set1_indices_tensor.flat<int64_t>();
        auto set1_shape_flat = set1_shape_tensor.flat<int64_t>();
        auto set2_indices_flat = set2_indices_tensor.flat<int64_t>();
        auto set2_shape_flat = set2_shape_tensor.flat<int64_t>();

        for (int i = 0; i < set1_indices_flat.size(); ++i) {
            set1_indices_flat(i) = std::abs(set1_indices_flat(i)) % 5;
        }
        for (int i = 0; i < set1_shape_flat.size(); ++i) {
            set1_shape_flat(i) = std::abs(set1_shape_flat(i)) % 10 + 1;
        }
        for (int i = 0; i < set2_indices_flat.size(); ++i) {
            set2_indices_flat(i) = std::abs(set2_indices_flat(i)) % 5;
        }
        for (int i = 0; i < set2_shape_flat.size(); ++i) {
            set2_shape_flat(i) = std::abs(set2_shape_flat(i)) % 10 + 1;
        }

        auto set1_indices_input = tensorflow::ops::Const(root, set1_indices_tensor);
        auto set1_values_input = tensorflow::ops::Const(root, set1_values_tensor);
        auto set1_shape_input = tensorflow::ops::Const(root, set1_shape_tensor);
        auto set2_indices_input = tensorflow::ops::Const(root, set2_indices_tensor);
        auto set2_values_input = tensorflow::ops::Const(root, set2_values_tensor);
        auto set2_shape_input = tensorflow::ops::Const(root, set2_shape_tensor);

        std::string set_operation = (offset < size && data[offset] % 2 == 0) ? "union" : "intersection";
        offset++;
        bool validate_indices = (offset < size) ? (data[offset] % 2 == 0) : true;

        // Use raw_ops instead of ops namespace
        auto sparse_set_op = tensorflow::ops::Raw(
            root.WithOpName("SparseToSparseSetOperation"),
            {set1_indices_input, set1_values_input, set1_shape_input, 
             set2_indices_input, set2_values_input, set2_shape_input},
            {tensorflow::DT_INT64, values_dtype, tensorflow::DT_INT64},
            {{"set_operation", set_operation}, {"validate_indices", validate_indices}}
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({sparse_set_op[0], sparse_set_op[1], sparse_set_op[2]}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
