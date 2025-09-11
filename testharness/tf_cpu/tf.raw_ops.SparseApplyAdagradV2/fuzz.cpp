#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/sparse_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include <iostream>
#include <cstring>
#include <vector>

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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        uint8_t var_rank = parseRank(data[offset++]);
        std::vector<int64_t> var_shape = parseShape(data, offset, size, var_rank);
        
        uint8_t grad_rank = parseRank(data[offset++]);
        std::vector<int64_t> grad_shape = parseShape(data, offset, size, grad_rank);
        
        uint8_t indices_rank = parseRank(data[offset++]);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        
        tensorflow::DataType indices_dtype = (data[offset++] % 2 == 0) ? tensorflow::DT_INT32 : tensorflow::DT_INT64;
        
        bool use_locking = (data[offset++] % 2 == 1);
        bool update_slots = (data[offset++] % 2 == 1);
        
        tensorflow::TensorShape var_tensor_shape(var_shape);
        tensorflow::TensorShape grad_tensor_shape(grad_shape);
        tensorflow::TensorShape indices_tensor_shape(indices_shape);
        tensorflow::TensorShape scalar_shape({});
        
        tensorflow::Tensor var_tensor(dtype, var_tensor_shape);
        tensorflow::Tensor accum_tensor(dtype, var_tensor_shape);
        tensorflow::Tensor lr_tensor(dtype, scalar_shape);
        tensorflow::Tensor epsilon_tensor(dtype, scalar_shape);
        tensorflow::Tensor grad_tensor(dtype, grad_tensor_shape);
        tensorflow::Tensor indices_tensor(indices_dtype, indices_tensor_shape);
        
        fillTensorWithDataByType(var_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(accum_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(lr_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(epsilon_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(grad_tensor, dtype, data, offset, size);
        
        if (indices_dtype == tensorflow::DT_INT32) {
            fillTensorWithData<int32_t>(indices_tensor, data, offset, size);
        } else {
            fillTensorWithData<int64_t>(indices_tensor, data, offset, size);
        }
        
        auto var_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto accum_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto lr_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto epsilon_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto grad_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto indices_placeholder = tensorflow::ops::Placeholder(root, indices_dtype);
        
        tensorflow::Output sparse_apply_adagrad_v2;
        
        tensorflow::NodeDef node_def;
        node_def.set_op("SparseApplyAdagradV2");
        node_def.set_name("SparseApplyAdagradV2");
        node_def.add_input(var_placeholder.node()->name());
        node_def.add_input(accum_placeholder.node()->name());
        node_def.add_input(lr_placeholder.node()->name());
        node_def.add_input(epsilon_placeholder.node()->name());
        node_def.add_input(grad_placeholder.node()->name());
        node_def.add_input(indices_placeholder.node()->name());
        
        auto attr = node_def.mutable_attr();
        (*attr)["T"].set_type(dtype);
        (*attr)["Tindices"].set_type(indices_dtype);
        (*attr)["use_locking"].set_b(use_locking);
        (*attr)["update_slots"].set_b(update_slots);
        
        tensorflow::Status status;
        auto op = root.AddNode(node_def, &status);
        sparse_apply_adagrad_v2 = tensorflow::Output(op);
        
        if (!status.ok()) {
            return -1;
        }
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({
            {var_placeholder, var_tensor},
            {accum_placeholder, accum_tensor},
            {lr_placeholder, lr_tensor},
            {epsilon_placeholder, epsilon_tensor},
            {grad_placeholder, grad_tensor},
            {indices_placeholder, indices_tensor}
        }, {sparse_apply_adagrad_v2}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
