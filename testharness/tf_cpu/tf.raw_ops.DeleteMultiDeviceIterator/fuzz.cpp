#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/dataset_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
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
    switch (selector % 21) {  
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
            dtype = tensorflow::DT_INT64;
            break;
        case 7:
            dtype = tensorflow::DT_BOOL;
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
        case 19:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 20:
            dtype = tensorflow::DT_RESOURCE;
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

REGISTER_OP("DeleteMultiDeviceIterator")
    .Input("multi_device_iterator: resource")
    .Input("iterators: N * resource")
    .Input("deleter: variant")
    .Attr("N: int >= 1")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs);

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::Tensor multi_device_iterator_tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
        
        if (offset < size) {
            uint8_t num_iterators = data[offset++] % 5 + 1;
            std::vector<tensorflow::Output> iterator_outputs;
            
            for (uint8_t i = 0; i < num_iterators && offset < size; ++i) {
                auto iterator_var = tensorflow::ops::Variable(root.WithOpName("iterator_" + std::to_string(i)), 
                                                            tensorflow::PartialTensorShape({}), 
                                                            tensorflow::DT_RESOURCE);
                iterator_outputs.push_back(iterator_var);
            }
            
            tensorflow::Tensor deleter_tensor(tensorflow::DT_VARIANT, tensorflow::TensorShape({}));
            tensorflow::Variant variant_value;
            deleter_tensor.scalar<tensorflow::Variant>()() = variant_value;
            
            auto multi_device_iterator_var = tensorflow::ops::Variable(root.WithOpName("multi_device_iterator"), 
                                                                     tensorflow::PartialTensorShape({}), 
                                                                     tensorflow::DT_RESOURCE);
            auto deleter_const = tensorflow::ops::Const(root, deleter_tensor);
            
            // Use raw operation node
            std::vector<tensorflow::Input> iterator_inputs;
            for (const auto& output : iterator_outputs) {
                iterator_inputs.push_back(output);
            }
            
            tensorflow::NodeDef delete_node_def;
            delete_node_def.set_name(root.UniqueName("DeleteMultiDeviceIterator"));
            delete_node_def.set_op("DeleteMultiDeviceIterator");
            
            // Add inputs to the NodeDef
            tensorflow::NodeDefBuilder node_builder(delete_node_def.name(), "DeleteMultiDeviceIterator");
            node_builder.Input(multi_device_iterator_var.node()->name(), 0, tensorflow::DT_RESOURCE);
            
            for (size_t i = 0; i < iterator_inputs.size(); ++i) {
                node_builder.Input(iterator_inputs[i].node()->name(), iterator_inputs[i].index(), tensorflow::DT_RESOURCE);
            }
            
            node_builder.Input(deleter_const.node()->name(), 0, tensorflow::DT_VARIANT);
            node_builder.Attr("N", static_cast<int>(iterator_inputs.size()));
            
            tensorflow::Status status = node_builder.Finalize(&delete_node_def);
            if (!status.ok()) {
                std::cout << "Error creating node: " << status.ToString() << std::endl;
                return -1;
            }
            
            tensorflow::Node* delete_node;
            status = root.graph()->AddNode(delete_node_def, &delete_node);
            if (!status.ok()) {
                std::cout << "Error adding node: " << status.ToString() << std::endl;
                return -1;
            }
            
            // Add edges
            root.graph()->AddEdge(multi_device_iterator_var.node(), 0, delete_node, 0);
            for (size_t i = 0; i < iterator_inputs.size(); ++i) {
                root.graph()->AddEdge(iterator_inputs[i].node(), iterator_inputs[i].index(), delete_node, i + 1);
            }
            root.graph()->AddEdge(deleter_const.node(), 0, delete_node, iterator_inputs.size() + 1);
            
            tensorflow::ClientSession session(root);
            
            std::cout << "Multi-device iterator created" << std::endl;
            std::cout << "Number of iterators: " << static_cast<int>(num_iterators) << std::endl;
            std::cout << "Deleter tensor created" << std::endl;
            
            status = session.Run({}, {});
            if (!status.ok()) {
                std::cout << "Error running session: " << status.ToString() << std::endl;
                return -1;
            }
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}