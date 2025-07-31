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
    switch (selector % 20) {
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
        case 17:
            dtype = tensorflow::DT_UINT32;
            break;
        case 18:
            dtype = tensorflow::DT_UINT64;
            break;
        default:
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
        
        uint8_t grad_rank = parseRank(data[offset++]);
        std::vector<int64_t> grad_shape = parseShape(data, offset, size, grad_rank);
        
        tensorflow::Tensor grad_tensor(dtype, tensorflow::TensorShape(grad_shape));
        fillTensorWithDataByType(grad_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor lr_tensor(dtype, tensorflow::TensorShape({}));
        fillTensorWithDataByType(lr_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor l1_tensor(dtype, tensorflow::TensorShape({}));
        fillTensorWithDataByType(l1_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor l2_tensor(dtype, tensorflow::TensorShape({}));
        fillTensorWithDataByType(l2_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor l2_shrinkage_tensor(dtype, tensorflow::TensorShape({}));
        fillTensorWithDataByType(l2_shrinkage_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor lr_power_tensor(dtype, tensorflow::TensorShape({}));
        fillTensorWithDataByType(lr_power_tensor, dtype, data, offset, size);
        
        auto var = tensorflow::ops::VarHandleOp(root, dtype, tensorflow::TensorShape(grad_shape));
        auto accum = tensorflow::ops::VarHandleOp(root, dtype, tensorflow::TensorShape(grad_shape));
        auto linear = tensorflow::ops::VarHandleOp(root, dtype, tensorflow::TensorShape(grad_shape));
        
        tensorflow::Tensor init_tensor(dtype, tensorflow::TensorShape(grad_shape));
        fillTensorWithDataByType(init_tensor, dtype, data, offset, size);
        
        auto var_init = tensorflow::ops::AssignVariableOp(root, var, tensorflow::ops::Const(root, init_tensor));
        auto accum_init = tensorflow::ops::AssignVariableOp(root, accum, tensorflow::ops::Const(root, init_tensor));
        auto linear_init = tensorflow::ops::AssignVariableOp(root, linear, tensorflow::ops::Const(root, init_tensor));
        
        bool use_locking = (data[offset % size] % 2) == 1;
        offset++;
        bool multiply_linear_by_lr = (data[offset % size] % 2) == 1;
        offset++;
        
        auto grad_input = tensorflow::ops::Const(root, grad_tensor);
        auto lr_input = tensorflow::ops::Const(root, lr_tensor);
        auto l1_input = tensorflow::ops::Const(root, l1_tensor);
        auto l2_input = tensorflow::ops::Const(root, l2_tensor);
        auto l2_shrinkage_input = tensorflow::ops::Const(root, l2_shrinkage_tensor);
        auto lr_power_input = tensorflow::ops::Const(root, lr_power_tensor);
        
        auto apply_ftrl = tensorflow::ops::ResourceApplyFtrlV2(
            root, var, accum, linear, grad_input, lr_input, l1_input, l2_input,
            l2_shrinkage_input, lr_power_input,
            tensorflow::ops::ResourceApplyFtrlV2::UseLocking(use_locking)
                .MultiplyLinearByLr(multiply_linear_by_lr));

        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Operation> init_ops = {var_init, accum_init, linear_init};
        tensorflow::Status init_status = session.Run({}, {}, init_ops, nullptr);
        if (!init_status.ok()) {
            return -1;
        }
        
        std::vector<tensorflow::Operation> ops = {apply_ftrl};
        tensorflow::Status status = session.Run({}, {}, ops, nullptr);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}