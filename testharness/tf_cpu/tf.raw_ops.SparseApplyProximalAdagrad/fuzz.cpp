#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/training_ops.h"
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
    switch (selector % 17) {
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
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 7:
            dtype = tensorflow::DT_INT64;
            break;
        case 8:
            dtype = tensorflow::DT_QINT8;
            break;
        case 9:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 10:
            dtype = tensorflow::DT_QINT32;
            break;
        case 11:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 12:
            dtype = tensorflow::DT_QINT16;
            break;
        case 13:
            dtype = tensorflow::DT_QUINT16;
            break;
        case 14:
            dtype = tensorflow::DT_UINT16;
            break;
        case 15:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 16:
            dtype = tensorflow::DT_HALF;
            break;
    }
    return dtype;
}

tensorflow::DataType parseIndicesDataType(uint8_t selector) {
    return (selector % 2 == 0) ? tensorflow::DT_INT32 : tensorflow::DT_INT64;
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
        case tensorflow::DT_UINT16:
            fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT32:
            fillTensorWithData<uint32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT64:
            fillTensorWithData<uint64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BFLOAT16:
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_COMPLEX64:
            fillTensorWithData<tensorflow::complex64>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_COMPLEX128:
            fillTensorWithData<tensorflow::complex128>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT8:
            fillTensorWithData<tensorflow::qint8>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QUINT8:
            fillTensorWithData<tensorflow::quint8>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT32:
            fillTensorWithData<tensorflow::qint32>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT16:
            fillTensorWithData<tensorflow::qint16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QUINT16:
            fillTensorWithData<tensorflow::quint16>(tensor, data, offset, total_size);
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
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        tensorflow::DataType indices_dtype = parseIndicesDataType(data[offset++]);
        
        uint8_t var_rank = parseRank(data[offset++]);
        uint8_t grad_rank = parseRank(data[offset++]);
        uint8_t indices_rank = std::min(static_cast<uint8_t>(1), parseRank(data[offset++]));
        
        std::vector<int64_t> var_shape = parseShape(data, offset, size, var_rank);
        std::vector<int64_t> grad_shape = parseShape(data, offset, size, grad_rank);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        
        if (var_shape.empty()) var_shape = {2, 3};
        if (grad_shape.empty()) grad_shape = {1, 3};
        if (indices_shape.empty()) indices_shape = {1};
        
        if (grad_shape.size() > 0 && var_shape.size() > 0) {
            grad_shape[0] = std::min(grad_shape[0], indices_shape[0]);
            for (size_t i = 1; i < grad_shape.size() && i < var_shape.size(); ++i) {
                grad_shape[i] = var_shape[i];
            }
        }
        
        tensorflow::TensorShape var_tensor_shape;
        for (auto dim : var_shape) {
            var_tensor_shape.AddDim(dim);
        }
        
        tensorflow::TensorShape accum_tensor_shape = var_tensor_shape;
        
        tensorflow::TensorShape grad_tensor_shape;
        for (auto dim : grad_shape) {
            grad_tensor_shape.AddDim(dim);
        }
        
        tensorflow::TensorShape indices_tensor_shape;
        for (auto dim : indices_shape) {
            indices_tensor_shape.AddDim(dim);
        }
        
        tensorflow::TensorShape scalar_shape;
        
        tensorflow::Tensor var_tensor(dtype, var_tensor_shape);
        tensorflow::Tensor accum_tensor(dtype, accum_tensor_shape);
        tensorflow::Tensor lr_tensor(dtype, scalar_shape);
        tensorflow::Tensor l1_tensor(dtype, scalar_shape);
        tensorflow::Tensor l2_tensor(dtype, scalar_shape);
        tensorflow::Tensor grad_tensor(dtype, grad_tensor_shape);
        tensorflow::Tensor indices_tensor(indices_dtype, indices_tensor_shape);
        
        fillTensorWithDataByType(var_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(accum_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(lr_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(l1_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(l2_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(grad_tensor, dtype, data, offset, size);
        
        if (indices_dtype == tensorflow::DT_INT32) {
            fillTensorWithData<int32_t>(indices_tensor, data, offset, size);
            auto indices_flat = indices_tensor.flat<int32_t>();
            for (int i = 0; i < indices_flat.size(); ++i) {
                indices_flat(i) = std::abs(indices_flat(i)) % static_cast<int32_t>(var_shape[0]);
            }
        } else {
            fillTensorWithData<int64_t>(indices_tensor, data, offset, size);
            auto indices_flat = indices_tensor.flat<int64_t>();
            for (int i = 0; i < indices_flat.size(); ++i) {
                indices_flat(i) = std::abs(indices_flat(i)) % static_cast<int64_t>(var_shape[0]);
            }
        }
        
        bool use_locking = (offset < size) ? (data[offset++] % 2 == 1) : false;
        
        auto var_input = tensorflow::ops::Variable(root, var_tensor.shape(), dtype);
        auto accum_input = tensorflow::ops::Variable(root, accum_tensor.shape(), dtype);
        auto lr_input = tensorflow::ops::Const(root, lr_tensor);
        auto l1_input = tensorflow::ops::Const(root, l1_tensor);
        auto l2_input = tensorflow::ops::Const(root, l2_tensor);
        auto grad_input = tensorflow::ops::Const(root, grad_tensor);
        auto indices_input = tensorflow::ops::Const(root, indices_tensor);
        
        auto assign_var = tensorflow::ops::Assign(root, var_input, tensorflow::ops::Const(root, var_tensor));
        auto assign_accum = tensorflow::ops::Assign(root, accum_input, tensorflow::ops::Const(root, accum_tensor));
        
        auto sparse_apply_proximal_adagrad = tensorflow::ops::SparseApplyProximalAdagrad(
            root, var_input, accum_input, lr_input, l1_input, l2_input, grad_input, indices_input,
            tensorflow::ops::SparseApplyProximalAdagrad::UseLocking(use_locking));
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> init_outputs;
        tensorflow::Status init_status = session.Run({assign_var, assign_accum}, &init_outputs);
        if (!init_status.ok()) {
            return -1;
        }
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({sparse_apply_proximal_adagrad}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}