#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/training_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

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
        
        if (size < 20) {
            return 0;
        }

        tensorflow::DataType dtype = parseDataType(data[offset++]);
        uint8_t rank = parseRank(data[offset++]);
        
        std::vector<int64_t> shape = parseShape(data, offset, size, rank);
        tensorflow::TensorShape tensor_shape(shape);
        
        tensorflow::Tensor var_tensor(dtype, tensor_shape);
        tensorflow::Tensor accum_tensor(dtype, tensor_shape);
        tensorflow::Tensor linear_tensor(dtype, tensor_shape);
        tensorflow::Tensor grad_tensor(dtype, tensor_shape);
        
        tensorflow::TensorShape scalar_shape({});
        tensorflow::Tensor lr_tensor(dtype, scalar_shape);
        tensorflow::Tensor l1_tensor(dtype, scalar_shape);
        tensorflow::Tensor l2_tensor(dtype, scalar_shape);
        tensorflow::Tensor l2_shrinkage_tensor(dtype, scalar_shape);
        tensorflow::Tensor lr_power_tensor(dtype, scalar_shape);
        
        fillTensorWithDataByType(var_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(accum_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(linear_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(grad_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(lr_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(l1_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(l2_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(l2_shrinkage_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(lr_power_tensor, dtype, data, offset, size);
        
        bool use_locking = (offset < size) ? (data[offset++] % 2 == 1) : false;
        bool multiply_linear_by_lr = (offset < size) ? (data[offset++] % 2 == 1) : false;
        
        std::cout << "var tensor shape: ";
        for (int i = 0; i < var_tensor.shape().dims(); ++i) {
            std::cout << var_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Data type: " << tensorflow::DataTypeString(dtype) << std::endl;
        std::cout << "Use locking: " << use_locking << std::endl;
        std::cout << "Multiply linear by lr: " << multiply_linear_by_lr << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto var_op = tensorflow::ops::Const(root, var_tensor);
        auto accum_op = tensorflow::ops::Const(root, accum_tensor);
        auto linear_op = tensorflow::ops::Const(root, linear_tensor);
        auto grad_op = tensorflow::ops::Const(root, grad_tensor);
        auto lr_op = tensorflow::ops::Const(root, lr_tensor);
        auto l1_op = tensorflow::ops::Const(root, l1_tensor);
        auto l2_op = tensorflow::ops::Const(root, l2_tensor);
        auto l2_shrinkage_op = tensorflow::ops::Const(root, l2_shrinkage_tensor);
        auto lr_power_op = tensorflow::ops::Const(root, lr_power_tensor);
        
        auto apply_ftrl_v2 = tensorflow::ops::ApplyFtrlV2(
            root,
            var_op,
            accum_op,
            linear_op,
            grad_op,
            lr_op,
            l1_op,
            l2_op,
            l2_shrinkage_op,
            lr_power_op,
            tensorflow::ops::ApplyFtrlV2::UseLocking(use_locking)
                .MultiplyLinearByLr(multiply_linear_by_lr)
        );
        
        tensorflow::GraphDef graph;
        TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        TF_RETURN_IF_ERROR(session->Create(graph));
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status run_status = session->Run({}, {apply_ftrl_v2.operation.name()}, {}, &outputs);
        
        if (!run_status.ok()) {
            std::cout << "Session run failed: " << run_status.ToString() << std::endl;
        } else {
            std::cout << "ApplyFtrlV2 operation executed successfully" << std::endl;
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}