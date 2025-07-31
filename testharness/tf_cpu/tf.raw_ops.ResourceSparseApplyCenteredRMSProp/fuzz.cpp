#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/training_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>

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
    switch (selector % 16) {
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
        default:
            dtype = tensorflow::DT_FLOAT;
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
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType lr_dtype = parseDataType(data[offset++]);
        tensorflow::DataType indices_dtype = parseIndicesDataType(data[offset++]);
        
        uint8_t var_rank = parseRank(data[offset++]);
        uint8_t indices_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> var_shape = parseShape(data, offset, size, var_rank);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        
        if (offset >= size) return 0;
        
        bool use_locking = (data[offset++] % 2) == 1;
        
        tensorflow::TensorShape var_tensor_shape(var_shape);
        tensorflow::TensorShape indices_tensor_shape(indices_shape);
        
        auto var_resource = tensorflow::ops::VarHandleOp(root, lr_dtype, var_tensor_shape);
        auto mg_resource = tensorflow::ops::VarHandleOp(root, lr_dtype, var_tensor_shape);
        auto ms_resource = tensorflow::ops::VarHandleOp(root, lr_dtype, var_tensor_shape);
        auto mom_resource = tensorflow::ops::VarHandleOp(root, lr_dtype, var_tensor_shape);
        
        tensorflow::Tensor var_init_tensor(lr_dtype, var_tensor_shape);
        fillTensorWithDataByType(var_init_tensor, lr_dtype, data, offset, size);
        auto var_init_const = tensorflow::ops::Const(root, var_init_tensor);
        
        tensorflow::Tensor mg_init_tensor(lr_dtype, var_tensor_shape);
        fillTensorWithDataByType(mg_init_tensor, lr_dtype, data, offset, size);
        auto mg_init_const = tensorflow::ops::Const(root, mg_init_tensor);
        
        tensorflow::Tensor ms_init_tensor(lr_dtype, var_tensor_shape);
        fillTensorWithDataByType(ms_init_tensor, lr_dtype, data, offset, size);
        auto ms_init_const = tensorflow::ops::Const(root, ms_init_tensor);
        
        tensorflow::Tensor mom_init_tensor(lr_dtype, var_tensor_shape);
        fillTensorWithDataByType(mom_init_tensor, lr_dtype, data, offset, size);
        auto mom_init_const = tensorflow::ops::Const(root, mom_init_tensor);
        
        auto var_assign = tensorflow::ops::AssignVariableOp(root, var_resource, var_init_const);
        auto mg_assign = tensorflow::ops::AssignVariableOp(root, mg_resource, mg_init_const);
        auto ms_assign = tensorflow::ops::AssignVariableOp(root, ms_resource, ms_init_const);
        auto mom_assign = tensorflow::ops::AssignVariableOp(root, mom_resource, mom_init_const);
        
        tensorflow::Tensor lr_tensor(lr_dtype, tensorflow::TensorShape({}));
        fillTensorWithDataByType(lr_tensor, lr_dtype, data, offset, size);
        auto lr_const = tensorflow::ops::Const(root, lr_tensor);
        
        tensorflow::Tensor rho_tensor(lr_dtype, tensorflow::TensorShape({}));
        fillTensorWithDataByType(rho_tensor, lr_dtype, data, offset, size);
        auto rho_const = tensorflow::ops::Const(root, rho_tensor);
        
        tensorflow::Tensor momentum_tensor(lr_dtype, tensorflow::TensorShape({}));
        fillTensorWithDataByType(momentum_tensor, lr_dtype, data, offset, size);
        auto momentum_const = tensorflow::ops::Const(root, momentum_tensor);
        
        tensorflow::Tensor epsilon_tensor(lr_dtype, tensorflow::TensorShape({}));
        fillTensorWithDataByType(epsilon_tensor, lr_dtype, data, offset, size);
        auto epsilon_const = tensorflow::ops::Const(root, epsilon_tensor);
        
        tensorflow::Tensor grad_tensor(lr_dtype, var_tensor_shape);
        fillTensorWithDataByType(grad_tensor, lr_dtype, data, offset, size);
        auto grad_const = tensorflow::ops::Const(root, grad_tensor);
        
        tensorflow::Tensor indices_tensor(indices_dtype, indices_tensor_shape);
        fillTensorWithDataByType(indices_tensor, indices_dtype, data, offset, size);
        auto indices_const = tensorflow::ops::Const(root, indices_tensor);
        
        auto sparse_apply_op = tensorflow::ops::ResourceSparseApplyCenteredRMSProp(
            root,
            var_resource,
            mg_resource,
            ms_resource,
            mom_resource,
            lr_const,
            rho_const,
            momentum_const,
            epsilon_const,
            grad_const,
            indices_const,
            tensorflow::ops::ResourceSparseApplyCenteredRMSProp::UseLocking(use_locking)
        );
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Operation> init_ops = {var_assign, mg_assign, ms_assign, mom_assign};
        tensorflow::Status init_status = session.Run({}, {}, init_ops, nullptr);
        if (!init_status.ok()) {
            return -1;
        }
        
        std::vector<tensorflow::Operation> run_ops = {sparse_apply_op};
        tensorflow::Status status = session.Run({}, {}, run_ops, nullptr);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}