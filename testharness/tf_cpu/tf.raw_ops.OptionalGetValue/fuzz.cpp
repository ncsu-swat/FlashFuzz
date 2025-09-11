#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/data_flow_ops.h"
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
            dtype = tensorflow::DT_STRING;
            break;
        case 7:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 8:
            dtype = tensorflow::DT_INT64;
            break;
        case 9:
            dtype = tensorflow::DT_BOOL;
            break;
        case 10:
            dtype = tensorflow::DT_QINT8;
            break;
        case 11:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 12:
            dtype = tensorflow::DT_QINT32;
            break;
        case 13:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 14:
            dtype = tensorflow::DT_QINT16;
            break;
        case 15:
            dtype = tensorflow::DT_QUINT16;
            break;
        case 16:
            dtype = tensorflow::DT_UINT16;
            break;
        case 17:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 18:
            dtype = tensorflow::DT_HALF;
            break;
        case 19:
            dtype = tensorflow::DT_UINT32;
            break;
        case 20:
            dtype = tensorflow::DT_UINT64;
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
        case tensorflow::DT_STRING: {
            auto flat = tensor.flat<tensorflow::tstring>();
            const size_t num_elements = flat.size();
            for (size_t i = 0; i < num_elements; ++i) {
                if (offset < total_size) {
                    uint8_t str_len = data[offset] % 10 + 1;
                    offset++;
                    std::string str;
                    for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                        str += static_cast<char>(data[offset]);
                        offset++;
                    }
                    flat(i) = str;
                } else {
                    flat(i) = "";
                }
            }
            break;
        }
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_output_types = (data[offset] % 3) + 1;
        offset++;

        std::vector<tensorflow::DataType> output_types;
        std::vector<tensorflow::TensorShape> output_shapes;
        
        for (uint8_t i = 0; i < num_output_types; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType dtype = parseDataType(data[offset]);
            output_types.push_back(dtype);
            offset++;
            
            if (offset >= size) break;
            uint8_t rank = parseRank(data[offset]);
            offset++;
            
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            output_shapes.push_back(tensorflow::TensorShape(shape));
        }
        
        if (output_types.empty()) {
            output_types.push_back(tensorflow::DT_FLOAT);
            output_shapes.push_back(tensorflow::TensorShape({1}));
        }

        tensorflow::Tensor optional_tensor(tensorflow::DT_VARIANT, tensorflow::TensorShape({}));
        
        // Create an optional using raw ops
        auto optional_none = tensorflow::ops::Const(root, optional_tensor);
        tensorflow::Output optional_op = optional_none;
        
        if (offset < size && data[offset] % 2 == 1) {
            std::vector<tensorflow::Output> values;
            for (size_t i = 0; i < output_types.size(); ++i) {
                if (offset >= size) break;
                
                tensorflow::Tensor value_tensor(output_types[i], output_shapes[i]);
                fillTensorWithDataByType(value_tensor, output_types[i], data, offset, size);
                
                auto const_op = tensorflow::ops::Const(root, value_tensor);
                values.push_back(const_op);
            }
            
            if (!values.empty()) {
                // Create an optional with values
                tensorflow::ops::Stack stack_op(root, values);
                optional_op = stack_op.output;
            }
        }
        
        // Use raw_ops.OptionalGetValue through direct node creation
        std::vector<tensorflow::Output> outputs;
        std::vector<tensorflow::DataType> dtypes(output_types);
        
        tensorflow::NodeDef node_def;
        node_def.set_name(root.UniqueName("OptionalGetValue"));
        node_def.set_op("OptionalGetValue");
        
        // Add input
        tensorflow::NodeDefBuilder builder(node_def.name(), "OptionalGetValue");
        builder.Input(tensorflow::NodeDefBuilder::NodeOut(optional_op.node()->name(), 0, 
                                                         tensorflow::DT_VARIANT));
        
        // Add attributes
        builder.Attr("output_types", dtypes);
        tensorflow::AttrValue shapes_attr;
        for (const auto& shape : output_shapes) {
            auto* shape_proto = shapes_attr.mutable_list()->add_shape();
            shape.AsProto(shape_proto);
        }
        builder.Attr("output_shapes", shapes_attr);
        
        tensorflow::Status status = builder.Finalize(&node_def);
        if (!status.ok()) {
            return -1;
        }
        
        tensorflow::Node* node;
        status = root.graph()->AddNode(node_def, &node);
        if (!status.ok()) {
            return -1;
        }
        
        root.graph()->AddEdge(optional_op.node(), 0, node, 0);
        
        for (int i = 0; i < output_types.size(); ++i) {
            outputs.push_back(tensorflow::Output(node, i));
        }
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> output_tensors;
        
        status = session.Run(outputs, &output_tensors);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
