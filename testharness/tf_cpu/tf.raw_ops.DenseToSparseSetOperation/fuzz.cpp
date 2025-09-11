#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/sparse_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <string>

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
    switch (selector % 7) {
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
        case 6:
            dtype = tensorflow::DT_STRING;
            break;
        default:
            dtype = tensorflow::DT_INT32;
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
                str += static_cast<char>(data[offset] % 26 + 'a');
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
            fillStringTensor(tensor, data, offset, total_size);
            break;
        default:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
    }
}

std::string parseSetOperation(uint8_t selector) {
    switch (selector % 4) {
        case 0: return "a-b";
        case 1: return "b-a";
        case 2: return "intersection";
        case 3: return "union";
        default: return "union";
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        uint8_t set1_rank = parseRank(data[offset++]);
        std::vector<int64_t> set1_shape = parseShape(data, offset, size, set1_rank);
        
        tensorflow::Tensor set1_tensor(dtype, tensorflow::TensorShape(set1_shape));
        fillTensorWithDataByType(set1_tensor, dtype, data, offset, size);
        
        uint8_t set2_nnz_byte = data[offset++];
        int64_t set2_nnz = (set2_nnz_byte % 10) + 1;
        
        std::vector<int64_t> set2_shape_vec = set1_shape;
        if (!set2_shape_vec.empty()) {
            set2_shape_vec.back() = (data[offset++] % 5) + 1;
        }
        
        tensorflow::Tensor set2_indices_tensor(tensorflow::DT_INT64, 
            tensorflow::TensorShape({set2_nnz, static_cast<int64_t>(set2_shape_vec.size())}));
        auto set2_indices_flat = set2_indices_tensor.flat<int64_t>();
        
        for (int64_t i = 0; i < set2_nnz; ++i) {
            for (size_t j = 0; j < set2_shape_vec.size(); ++j) {
                if (offset < size) {
                    int64_t idx_val;
                    if (offset + sizeof(int64_t) <= size) {
                        std::memcpy(&idx_val, data + offset, sizeof(int64_t));
                        offset += sizeof(int64_t);
                    } else {
                        idx_val = data[offset++];
                    }
                    set2_indices_flat(i * set2_shape_vec.size() + j) = 
                        std::abs(idx_val) % set2_shape_vec[j];
                } else {
                    set2_indices_flat(i * set2_shape_vec.size() + j) = 0;
                }
            }
        }
        
        tensorflow::Tensor set2_values_tensor(dtype, tensorflow::TensorShape({set2_nnz}));
        fillTensorWithDataByType(set2_values_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor set2_shape_tensor(tensorflow::DT_INT64, 
            tensorflow::TensorShape({static_cast<int64_t>(set2_shape_vec.size())}));
        auto set2_shape_flat = set2_shape_tensor.flat<int64_t>();
        for (size_t i = 0; i < set2_shape_vec.size(); ++i) {
            set2_shape_flat(i) = set2_shape_vec[i];
        }
        
        std::string set_operation = parseSetOperation(data[offset++]);
        bool validate_indices = (data[offset++] % 2) == 0;
        
        auto set1_op = tensorflow::ops::Const(root, set1_tensor);
        auto set2_indices_op = tensorflow::ops::Const(root, set2_indices_tensor);
        auto set2_values_op = tensorflow::ops::Const(root, set2_values_tensor);
        auto set2_shape_op = tensorflow::ops::Const(root, set2_shape_tensor);
        
        // Use raw_ops namespace for DenseToSparseSetOperation
        auto dense_to_sparse_set_op = tensorflow::ops::Raw::DenseToSparseSetOperation(
            root, set1_op, set2_indices_op, set2_values_op, set2_shape_op, 
            set_operation, validate_indices);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({
            dense_to_sparse_set_op.result_indices,
            dense_to_sparse_set_op.result_values,
            dense_to_sparse_set_op.result_shape}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
