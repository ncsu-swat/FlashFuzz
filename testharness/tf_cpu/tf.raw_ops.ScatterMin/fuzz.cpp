#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
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
    switch (selector % 6) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 2:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 3:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 4:
            dtype = tensorflow::DT_INT32;
            break;
        case 5:
            dtype = tensorflow::DT_INT64;
            break;
    }
    return dtype;
}

tensorflow::DataType parseIndicesDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
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
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BFLOAT16:
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
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
        tensorflow::DataType ref_dtype = parseDataType(data[offset++]);
        tensorflow::DataType indices_dtype = parseIndicesDataType(data[offset++]);
        
        uint8_t ref_rank = parseRank(data[offset++]);
        uint8_t indices_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> ref_shape = parseShape(data, offset, size, ref_rank);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        
        if (ref_shape.empty()) {
            ref_shape = {1};
        }
        if (indices_shape.empty()) {
            indices_shape = {1};
        }
        
        std::vector<int64_t> updates_shape = indices_shape;
        if (ref_shape.size() > 1) {
            for (size_t i = 1; i < ref_shape.size(); ++i) {
                updates_shape.push_back(ref_shape[i]);
            }
        }
        
        tensorflow::TensorShape ref_tensor_shape(ref_shape);
        tensorflow::TensorShape indices_tensor_shape(indices_shape);
        tensorflow::TensorShape updates_tensor_shape(updates_shape);
        
        tensorflow::Tensor ref_tensor(ref_dtype, ref_tensor_shape);
        tensorflow::Tensor indices_tensor(indices_dtype, indices_tensor_shape);
        tensorflow::Tensor updates_tensor(ref_dtype, updates_tensor_shape);
        
        fillTensorWithDataByType(ref_tensor, ref_dtype, data, offset, size);
        fillTensorWithDataByType(indices_tensor, indices_dtype, data, offset, size);
        fillTensorWithDataByType(updates_tensor, ref_dtype, data, offset, size);
        
        if (indices_dtype == tensorflow::DT_INT32) {
            auto indices_flat = indices_tensor.flat<int32_t>();
            for (int i = 0; i < indices_flat.size(); ++i) {
                indices_flat(i) = std::abs(indices_flat(i)) % static_cast<int32_t>(ref_shape[0]);
            }
        } else {
            auto indices_flat = indices_tensor.flat<int64_t>();
            for (int i = 0; i < indices_flat.size(); ++i) {
                indices_flat(i) = std::abs(indices_flat(i)) % ref_shape[0];
            }
        }
        
        auto ref_var = tensorflow::ops::Variable(root, ref_tensor_shape, ref_dtype);
        auto assign_ref = tensorflow::ops::Assign(root, ref_var, tensorflow::ops::Const(root, ref_tensor));
        
        bool use_locking = (data[offset % size] % 2) == 1;
        
        auto scatter_min = tensorflow::ops::ScatterMin(
            root, 
            ref_var, 
            tensorflow::ops::Const(root, indices_tensor),
            tensorflow::ops::Const(root, updates_tensor),
            tensorflow::ops::ScatterMin::UseLocking(use_locking)
        );
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> assign_outputs;
        tensorflow::Status assign_status = session.Run({assign_ref}, &assign_outputs);
        if (!assign_status.ok()) {
            return -1;
        }
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({scatter_min}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
