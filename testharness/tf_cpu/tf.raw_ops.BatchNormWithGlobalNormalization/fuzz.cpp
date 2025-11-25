#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/node_builder.h"
#include <iostream>
#include <cstring>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 4
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    switch (selector % 3) {
        case 0:
            return tensorflow::DT_FLOAT;
        case 1:
            return tensorflow::DT_DOUBLE;
        case 2:
            return tensorflow::DT_HALF;
        default:
            return tensorflow::DT_FLOAT;
    }
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
            return;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        uint8_t rank = parseRank(data[offset++]);
        std::vector<int64_t> t_shape = parseShape(data, offset, size, rank);
        
        if (t_shape.size() != 4) {
            return 0;
        }
        
        int64_t last_dim = t_shape[3];
        std::vector<int64_t> vec_shape = {last_dim};
        
        tensorflow::TensorShape t_tensor_shape(t_shape);
        tensorflow::TensorShape vec_tensor_shape(vec_shape);
        
        tensorflow::Tensor t_tensor(dtype, t_tensor_shape);
        tensorflow::Tensor m_tensor(dtype, vec_tensor_shape);
        tensorflow::Tensor v_tensor(dtype, vec_tensor_shape);
        tensorflow::Tensor beta_tensor(dtype, vec_tensor_shape);
        tensorflow::Tensor gamma_tensor(dtype, vec_tensor_shape);
        
        fillTensorWithDataByType(t_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(m_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(v_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(beta_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(gamma_tensor, dtype, data, offset, size);
        
        float variance_epsilon = 1e-5f;
        if (offset < size) {
            std::memcpy(&variance_epsilon, data + offset, std::min(sizeof(float), size - offset));
            offset += sizeof(float);
            variance_epsilon = std::abs(variance_epsilon);
            if (variance_epsilon == 0.0f) variance_epsilon = 1e-5f;
        }
        
        bool scale_after_normalization = false;
        if (offset < size) {
            scale_after_normalization = (data[offset] % 2) == 1;
        }
        
        auto t_input = tensorflow::ops::Const(root, t_tensor);
        auto m_input = tensorflow::ops::Const(root, m_tensor);
        auto v_input = tensorflow::ops::Const(root, v_tensor);
        auto beta_input = tensorflow::ops::Const(root, beta_tensor);
        auto gamma_input = tensorflow::ops::Const(root, gamma_tensor);
        
        tensorflow::Node* batch_norm_node = nullptr;
        tensorflow::Status status = tensorflow::NodeBuilder(
                                        root.GetUniqueNameForOp("BatchNormWithGlobalNormalization"),
                                        "BatchNormWithGlobalNormalization")
                                        .Input(t_input.node())
                                        .Input(m_input.node())
                                        .Input(v_input.node())
                                        .Input(beta_input.node())
                                        .Input(gamma_input.node())
                                        .Attr("variance_epsilon", variance_epsilon)
                                        .Attr("scale_after_normalization", scale_after_normalization)
                                        .Attr("T", dtype)
                                        .Finalize(root.graph(), &batch_norm_node);
        if (!status.ok()) {
            tf_fuzzer_utils::logError("Failed to create BatchNormWithGlobalNormalization op: " + status.ToString(), data, size);
            return -1;
        }
        
        tensorflow::Output batch_norm_op(batch_norm_node, 0);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({batch_norm_op}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
