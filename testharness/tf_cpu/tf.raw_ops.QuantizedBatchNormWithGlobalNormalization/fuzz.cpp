#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cstring>
#include <vector>
#include <iostream>

#define MAX_RANK 4
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << message << std::endl;
    }
}

tensorflow::DataType parseQuantizedDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 5) {
        case 0:
            dtype = tensorflow::DT_QINT8;
            break;
        case 1:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 2:
            dtype = tensorflow::DT_QINT32;
            break;
        case 3:
            dtype = tensorflow::DT_QINT16;
            break;
        case 4:
            dtype = tensorflow::DT_QUINT16;
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
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType input_dtype = parseQuantizedDataType(data[offset++]);
        tensorflow::DataType out_dtype = parseQuantizedDataType(data[offset++]);
        
        uint8_t t_rank = parseRank(data[offset++]);
        std::vector<int64_t> t_shape = parseShape(data, offset, size, t_rank);
        
        if (t_shape.size() != 4) {
            t_shape = {2, 3, 4, 5};
        }
        
        int64_t last_dim = t_shape.back();
        std::vector<int64_t> param_shape = {last_dim};
        
        tensorflow::Tensor t_tensor(input_dtype, tensorflow::TensorShape(t_shape));
        fillTensorWithDataByType(t_tensor, input_dtype, data, offset, size);
        
        tensorflow::Tensor t_min_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor t_max_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        fillTensorWithDataByType(t_min_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(t_max_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::Tensor m_tensor(input_dtype, tensorflow::TensorShape(param_shape));
        fillTensorWithDataByType(m_tensor, input_dtype, data, offset, size);
        
        tensorflow::Tensor m_min_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor m_max_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        fillTensorWithDataByType(m_min_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(m_max_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::Tensor v_tensor(input_dtype, tensorflow::TensorShape(param_shape));
        fillTensorWithDataByType(v_tensor, input_dtype, data, offset, size);
        
        tensorflow::Tensor v_min_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor v_max_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        fillTensorWithDataByType(v_min_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(v_max_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::Tensor beta_tensor(input_dtype, tensorflow::TensorShape(param_shape));
        fillTensorWithDataByType(beta_tensor, input_dtype, data, offset, size);
        
        tensorflow::Tensor beta_min_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor beta_max_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        fillTensorWithDataByType(beta_min_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(beta_max_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        tensorflow::Tensor gamma_tensor(input_dtype, tensorflow::TensorShape(param_shape));
        fillTensorWithDataByType(gamma_tensor, input_dtype, data, offset, size);
        
        tensorflow::Tensor gamma_min_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor gamma_max_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        fillTensorWithDataByType(gamma_min_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(gamma_max_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        float variance_epsilon = 1e-5f;
        if (offset < size) {
            std::memcpy(&variance_epsilon, data + offset, std::min(sizeof(float), size - offset));
            variance_epsilon = std::abs(variance_epsilon);
            if (variance_epsilon == 0.0f) variance_epsilon = 1e-5f;
        }
        
        bool scale_after_normalization = (offset < size) ? (data[offset] % 2 == 1) : true;
        
        auto t_input = tensorflow::ops::Const(root, t_tensor);
        auto t_min_input = tensorflow::ops::Const(root, t_min_tensor);
        auto t_max_input = tensorflow::ops::Const(root, t_max_tensor);
        auto m_input = tensorflow::ops::Const(root, m_tensor);
        auto m_min_input = tensorflow::ops::Const(root, m_min_tensor);
        auto m_max_input = tensorflow::ops::Const(root, m_max_tensor);
        auto v_input = tensorflow::ops::Const(root, v_tensor);
        auto v_min_input = tensorflow::ops::Const(root, v_min_tensor);
        auto v_max_input = tensorflow::ops::Const(root, v_max_tensor);
        auto beta_input = tensorflow::ops::Const(root, beta_tensor);
        auto beta_min_input = tensorflow::ops::Const(root, beta_min_tensor);
        auto beta_max_input = tensorflow::ops::Const(root, beta_max_tensor);
        auto gamma_input = tensorflow::ops::Const(root, gamma_tensor);
        auto gamma_min_input = tensorflow::ops::Const(root, gamma_min_tensor);
        auto gamma_max_input = tensorflow::ops::Const(root, gamma_max_tensor);
        
        tensorflow::Output result;
        tensorflow::Output result_min;
        tensorflow::Output result_max;
        
        tensorflow::NodeDef def;
        def.set_op("QuantizedBatchNormWithGlobalNormalization");
        def.set_device("/cpu:0");
        
        tensorflow::NodeDefBuilder builder("quantized_batch_norm", "QuantizedBatchNormWithGlobalNormalization");
        builder.Input(tensorflow::NodeDefBuilder::NodeOut(t_input.node()->name(), 0, input_dtype))
               .Input(tensorflow::NodeDefBuilder::NodeOut(t_min_input.node()->name(), 0, tensorflow::DT_FLOAT))
               .Input(tensorflow::NodeDefBuilder::NodeOut(t_max_input.node()->name(), 0, tensorflow::DT_FLOAT))
               .Input(tensorflow::NodeDefBuilder::NodeOut(m_input.node()->name(), 0, input_dtype))
               .Input(tensorflow::NodeDefBuilder::NodeOut(m_min_input.node()->name(), 0, tensorflow::DT_FLOAT))
               .Input(tensorflow::NodeDefBuilder::NodeOut(m_max_input.node()->name(), 0, tensorflow::DT_FLOAT))
               .Input(tensorflow::NodeDefBuilder::NodeOut(v_input.node()->name(), 0, input_dtype))
               .Input(tensorflow::NodeDefBuilder::NodeOut(v_min_input.node()->name(), 0, tensorflow::DT_FLOAT))
               .Input(tensorflow::NodeDefBuilder::NodeOut(v_max_input.node()->name(), 0, tensorflow::DT_FLOAT))
               .Input(tensorflow::NodeDefBuilder::NodeOut(beta_input.node()->name(), 0, input_dtype))
               .Input(tensorflow::NodeDefBuilder::NodeOut(beta_min_input.node()->name(), 0, tensorflow::DT_FLOAT))
               .Input(tensorflow::NodeDefBuilder::NodeOut(beta_max_input.node()->name(), 0, tensorflow::DT_FLOAT))
               .Input(tensorflow::NodeDefBuilder::NodeOut(gamma_input.node()->name(), 0, input_dtype))
               .Input(tensorflow::NodeDefBuilder::NodeOut(gamma_min_input.node()->name(), 0, tensorflow::DT_FLOAT))
               .Input(tensorflow::NodeDefBuilder::NodeOut(gamma_max_input.node()->name(), 0, tensorflow::DT_FLOAT))
               .Attr("T", input_dtype)
               .Attr("out_type", out_dtype)
               .Attr("variance_epsilon", variance_epsilon)
               .Attr("scale_after_normalization", scale_after_normalization);
        
        tensorflow::Status status = builder.Finalize(&def);
        if (!status.ok()) {
            return -1;
        }
        
        tensorflow::Node* node;
        status = root.graph()->AddNode(def, &node);
        if (!status.ok()) {
            return -1;
        }
        
        result = tensorflow::Output(node, 0);
        result_min = tensorflow::Output(node, 1);
        result_max = tensorflow::Output(node, 2);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({result, result_min, result_max}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}