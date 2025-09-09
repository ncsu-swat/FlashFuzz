#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/framework/node_def_util.h>
#include <tensorflow/core/kernels/training_ops.h>

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

        tensorflow::DataType var_dtype = parseDataType(data[offset++]);
        tensorflow::DataType indices_dtype = parseIndicesDataType(data[offset++]);
        
        uint8_t var_rank = parseRank(data[offset++]);
        uint8_t indices_rank = 1;
        
        std::vector<int64_t> var_shape = parseShape(data, offset, size, var_rank);
        
        if (offset >= size) return 0;
        
        int64_t num_indices = 1 + (data[offset++] % 5);
        std::vector<int64_t> indices_shape = {num_indices};
        
        std::vector<int64_t> grad_shape = var_shape;
        if (!grad_shape.empty()) {
            grad_shape[0] = num_indices;
        }
        
        std::vector<int64_t> scalar_shape = {};
        
        tensorflow::TensorShape var_tensor_shape(var_shape);
        tensorflow::TensorShape grad_tensor_shape(grad_shape);
        tensorflow::TensorShape indices_tensor_shape(indices_shape);
        tensorflow::TensorShape scalar_tensor_shape(scalar_shape);
        
        tensorflow::Tensor var_tensor(var_dtype, var_tensor_shape);
        tensorflow::Tensor gradient_accumulator_tensor(var_dtype, var_tensor_shape);
        tensorflow::Tensor gradient_squared_accumulator_tensor(var_dtype, var_tensor_shape);
        tensorflow::Tensor grad_tensor(var_dtype, grad_tensor_shape);
        tensorflow::Tensor indices_tensor(indices_dtype, indices_tensor_shape);
        tensorflow::Tensor lr_tensor(var_dtype, scalar_tensor_shape);
        tensorflow::Tensor l1_tensor(var_dtype, scalar_tensor_shape);
        tensorflow::Tensor l2_tensor(var_dtype, scalar_tensor_shape);
        tensorflow::Tensor global_step_tensor(tensorflow::DT_INT64, scalar_tensor_shape);
        
        fillTensorWithDataByType(var_tensor, var_dtype, data, offset, size);
        fillTensorWithDataByType(gradient_accumulator_tensor, var_dtype, data, offset, size);
        fillTensorWithDataByType(gradient_squared_accumulator_tensor, var_dtype, data, offset, size);
        fillTensorWithDataByType(grad_tensor, var_dtype, data, offset, size);
        fillTensorWithDataByType(indices_tensor, indices_dtype, data, offset, size);
        fillTensorWithDataByType(lr_tensor, var_dtype, data, offset, size);
        fillTensorWithDataByType(l1_tensor, var_dtype, data, offset, size);
        fillTensorWithDataByType(l2_tensor, var_dtype, data, offset, size);
        fillTensorWithDataByType(global_step_tensor, tensorflow::DT_INT64, data, offset, size);
        
        if (indices_dtype == tensorflow::DT_INT32) {
            auto indices_flat = indices_tensor.flat<int32_t>();
            for (int i = 0; i < indices_flat.size(); ++i) {
                indices_flat(i) = std::abs(indices_flat(i)) % static_cast<int32_t>(var_shape.empty() ? 1 : var_shape[0]);
            }
        } else {
            auto indices_flat = indices_tensor.flat<int64_t>();
            for (int i = 0; i < indices_flat.size(); ++i) {
                indices_flat(i) = std::abs(indices_flat(i)) % (var_shape.empty() ? 1 : var_shape[0]);
            }
        }
        
        bool use_locking = (offset < size) ? (data[offset++] % 2 == 1) : false;
        
        std::cout << "var shape: ";
        for (auto dim : var_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "grad shape: ";
        for (auto dim : grad_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "indices shape: ";
        for (auto dim : indices_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "var_dtype: " << var_dtype << std::endl;
        std::cout << "indices_dtype: " << indices_dtype << std::endl;
        std::cout << "use_locking: " << use_locking << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto var_placeholder = tensorflow::ops::Placeholder(root.WithOpName("var"), var_dtype, 
            tensorflow::ops::Placeholder::Shape(var_tensor_shape));
        auto gradient_accumulator_placeholder = tensorflow::ops::Placeholder(root.WithOpName("gradient_accumulator"), var_dtype,
            tensorflow::ops::Placeholder::Shape(var_tensor_shape));
        auto gradient_squared_accumulator_placeholder = tensorflow::ops::Placeholder(root.WithOpName("gradient_squared_accumulator"), var_dtype,
            tensorflow::ops::Placeholder::Shape(var_tensor_shape));
        auto grad_placeholder = tensorflow::ops::Placeholder(root.WithOpName("grad"), var_dtype,
            tensorflow::ops::Placeholder::Shape(grad_tensor_shape));
        auto indices_placeholder = tensorflow::ops::Placeholder(root.WithOpName("indices"), indices_dtype,
            tensorflow::ops::Placeholder::Shape(indices_tensor_shape));
        auto lr_placeholder = tensorflow::ops::Placeholder(root.WithOpName("lr"), var_dtype,
            tensorflow::ops::Placeholder::Shape(scalar_tensor_shape));
        auto l1_placeholder = tensorflow::ops::Placeholder(root.WithOpName("l1"), var_dtype,
            tensorflow::ops::Placeholder::Shape(scalar_tensor_shape));
        auto l2_placeholder = tensorflow::ops::Placeholder(root.WithOpName("l2"), var_dtype,
            tensorflow::ops::Placeholder::Shape(scalar_tensor_shape));
        auto global_step_placeholder = tensorflow::ops::Placeholder(root.WithOpName("global_step"), tensorflow::DT_INT64,
            tensorflow::ops::Placeholder::Shape(scalar_tensor_shape));
        
        tensorflow::NodeDef node_def;
        node_def.set_op("SparseApplyAdagradDA");
        node_def.set_name("sparse_apply_adagrad_da");
        
        node_def.add_input("var");
        node_def.add_input("gradient_accumulator");
        node_def.add_input("gradient_squared_accumulator");
        node_def.add_input("grad");
        node_def.add_input("indices");
        node_def.add_input("lr");
        node_def.add_input("l1");
        node_def.add_input("l2");
        node_def.add_input("global_step");
        
        tensorflow::AddNodeAttr("T", var_dtype, &node_def);
        tensorflow::AddNodeAttr("Tindices", indices_dtype, &node_def);
        tensorflow::AddNodeAttr("use_locking", use_locking, &node_def);
        
        tensorflow::Status status = root.graph()->AddNode(node_def, nullptr);
        if (!status.ok()) {
            std::cout << "Failed to add node: " << status.ToString() << std::endl;
            return 0;
        }
        
        tensorflow::GraphDef graph_def;
        status = root.ToGraphDef(&graph_def);
        if (!status.ok()) {
            std::cout << "Failed to convert to GraphDef: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = session->Create(graph_def);
        if (!status.ok()) {
            std::cout << "Failed to create session: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"var", var_tensor},
            {"gradient_accumulator", gradient_accumulator_tensor},
            {"gradient_squared_accumulator", gradient_squared_accumulator_tensor},
            {"grad", grad_tensor},
            {"indices", indices_tensor},
            {"lr", lr_tensor},
            {"l1", l1_tensor},
            {"l2", l2_tensor},
            {"global_step", global_step_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"sparse_apply_adagrad_da"}, {}, &outputs);
        if (!status.ok()) {
            std::cout << "Failed to run session: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::cout << "Operation completed successfully" << std::endl;
        if (!outputs.empty()) {
            std::cout << "Output tensor shape: " << outputs[0].shape().DebugString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}