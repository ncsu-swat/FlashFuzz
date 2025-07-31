#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
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
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType ref_dtype = parseDataType(data[offset++]);
        uint8_t ref_rank = parseRank(data[offset++]);
        std::vector<int64_t> ref_shape = parseShape(data, offset, size, ref_rank);
        
        if (offset >= size) return 0;
        
        uint8_t indices_rank = parseRank(data[offset++]);
        if (indices_rank == 0) indices_rank = 1;
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        
        if (offset >= size) return 0;
        
        int64_t K = 1;
        if (indices_rank > 0 && !indices_shape.empty()) {
            K = indices_shape.back();
            if (K <= 0) K = 1;
            if (ref_rank > 0 && K > ref_rank) K = ref_rank;
        }
        
        std::vector<int64_t> updates_shape;
        if (indices_rank > 1) {
            for (int i = 0; i < indices_rank - 1; ++i) {
                updates_shape.push_back(indices_shape[i]);
            }
        }
        if (ref_rank > 0 && K < ref_rank) {
            for (int64_t i = K; i < ref_rank; ++i) {
                updates_shape.push_back(ref_shape[i]);
            }
        }
        if (updates_shape.empty()) {
            updates_shape.push_back(1);
        }
        
        tensorflow::TensorShape ref_tensor_shape(ref_shape);
        tensorflow::TensorShape indices_tensor_shape(indices_shape);
        tensorflow::TensorShape updates_tensor_shape(updates_shape);
        
        auto var_handle = tensorflow::ops::VarHandleOp(root, ref_dtype, ref_tensor_shape);
        
        tensorflow::Tensor init_tensor(ref_dtype, ref_tensor_shape);
        fillTensorWithDataByType(init_tensor, ref_dtype, data, offset, size);
        
        auto init_op = tensorflow::ops::AssignVariableOp(root, var_handle, init_tensor);
        
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, indices_tensor_shape);
        fillTensorWithData<int32_t>(indices_tensor, data, offset, size);
        
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < indices_flat.size(); ++i) {
            int32_t val = indices_flat(i);
            if (ref_rank > 0 && !ref_shape.empty()) {
                int dim_idx = i % ref_rank;
                if (ref_shape[dim_idx] > 0) {
                    val = std::abs(val) % static_cast<int32_t>(ref_shape[dim_idx]);
                } else {
                    val = 0;
                }
            } else {
                val = 0;
            }
            indices_flat(i) = val;
        }
        
        tensorflow::Tensor updates_tensor(ref_dtype, updates_tensor_shape);
        fillTensorWithDataByType(updates_tensor, ref_dtype, data, offset, size);
        
        bool use_locking = (data[offset % size] % 2) == 0;
        
        auto scatter_op = tensorflow::ops::ResourceScatterNdAdd(
            root, var_handle, indices_tensor, updates_tensor,
            tensorflow::ops::ResourceScatterNdAdd::UseLocking(use_locking)
        );
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Operation> ops_to_run = {init_op};
        tensorflow::Status init_status = session.Run(ops_to_run, nullptr);
        if (!init_status.ok()) {
            return -1;
        }
        
        ops_to_run = {scatter_op.operation};
        tensorflow::Status status = session.Run(ops_to_run, nullptr);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}