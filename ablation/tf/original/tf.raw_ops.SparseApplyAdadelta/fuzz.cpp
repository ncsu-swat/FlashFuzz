#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/sparse_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tensorflow/core/platform/types.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

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
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType dtype = parseDataType(data[offset++]);
        uint8_t var_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> var_shape = parseShape(data, offset, size, var_rank);
        tensorflow::TensorShape var_tensor_shape(var_shape);
        
        tensorflow::Tensor var_tensor(dtype, var_tensor_shape);
        fillTensorWithDataByType(var_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor accum_tensor(dtype, var_tensor_shape);
        fillTensorWithDataByType(accum_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor accum_update_tensor(dtype, var_tensor_shape);
        fillTensorWithDataByType(accum_update_tensor, dtype, data, offset, size);
        
        tensorflow::TensorShape scalar_shape({});
        tensorflow::Tensor lr_tensor(dtype, scalar_shape);
        fillTensorWithDataByType(lr_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor rho_tensor(dtype, scalar_shape);
        fillTensorWithDataByType(rho_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor epsilon_tensor(dtype, scalar_shape);
        fillTensorWithDataByType(epsilon_tensor, dtype, data, offset, size);
        
        tensorflow::Tensor grad_tensor(dtype, var_tensor_shape);
        fillTensorWithDataByType(grad_tensor, dtype, data, offset, size);
        
        tensorflow::DataType indices_dtype = (offset < size && data[offset] % 2 == 0) ? 
            tensorflow::DT_INT32 : tensorflow::DT_INT64;
        offset++;
        
        uint8_t indices_rank = 1;
        std::vector<int64_t> indices_shape = {std::min(static_cast<int64_t>(5), var_shape.empty() ? 1 : var_shape[0])};
        tensorflow::TensorShape indices_tensor_shape(indices_shape);
        
        tensorflow::Tensor indices_tensor(indices_dtype, indices_tensor_shape);
        if (indices_dtype == tensorflow::DT_INT32) {
            fillTensorWithData<int32_t>(indices_tensor, data, offset, size);
        } else {
            fillTensorWithData<int64_t>(indices_tensor, data, offset, size);
        }
        
        bool use_locking = (offset < size) ? (data[offset++] % 2 == 1) : false;
        
        std::cout << "var tensor shape: ";
        for (int i = 0; i < var_tensor.shape().dims(); ++i) {
            std::cout << var_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "indices tensor shape: ";
        for (int i = 0; i < indices_tensor.shape().dims(); ++i) {
            std::cout << indices_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "use_locking: " << use_locking << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto var_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto accum_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto accum_update_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto lr_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto rho_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto epsilon_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto grad_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto indices_placeholder = tensorflow::ops::Placeholder(root, indices_dtype);
        
        auto sparse_apply_adadelta = tensorflow::ops::SparseApplyAdadelta(
            root, var_placeholder, accum_placeholder, accum_update_placeholder,
            lr_placeholder, rho_placeholder, epsilon_placeholder, grad_placeholder,
            indices_placeholder, tensorflow::ops::SparseApplyAdadelta::UseLocking(use_locking));
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({
            {var_placeholder, var_tensor},
            {accum_placeholder, accum_tensor},
            {accum_update_placeholder, accum_update_tensor},
            {lr_placeholder, lr_tensor},
            {rho_placeholder, rho_tensor},
            {epsilon_placeholder, epsilon_tensor},
            {grad_placeholder, grad_tensor},
            {indices_placeholder, indices_tensor}
        }, {sparse_apply_adadelta}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::cout << "Operation completed successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}