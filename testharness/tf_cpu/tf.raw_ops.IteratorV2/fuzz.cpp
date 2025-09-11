#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/dataset_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

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

std::string parseString(const uint8_t* data, size_t& offset, size_t total_size) {
    if (offset >= total_size) {
        return "default";
    }
    
    uint8_t str_len = data[offset] % 32 + 1;
    offset++;
    
    std::string result;
    for (uint8_t i = 0; i < str_len && offset < total_size; ++i) {
        char c = static_cast<char>(data[offset] % 94 + 33);
        result += c;
        offset++;
    }
    
    if (result.empty()) {
        result = "default";
    }
    
    return result;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) {
        return 0;
    }
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        std::string shared_name = parseString(data, offset, size);
        std::string container = parseString(data, offset, size);
        
        if (offset >= size) {
            return 0;
        }
        
        uint8_t num_types = (data[offset] % 5) + 1;
        offset++;
        
        std::vector<tensorflow::DataType> output_types;
        std::vector<tensorflow::TensorShape> output_shapes;
        
        for (uint8_t i = 0; i < num_types && offset < size; ++i) {
            tensorflow::DataType dtype = parseDataType(data[offset]);
            offset++;
            
            if (offset >= size) {
                break;
            }
            
            uint8_t rank = parseRank(data[offset]);
            offset++;
            
            std::vector<int64_t> shape_dims = parseShape(data, offset, size, rank);
            
            output_types.push_back(dtype);
            
            tensorflow::TensorShape shape;
            for (int64_t dim : shape_dims) {
                shape.AddDim(dim);
            }
            output_shapes.push_back(shape);
        }
        
        if (output_types.empty()) {
            output_types.push_back(tensorflow::DT_FLOAT);
            output_shapes.push_back(tensorflow::TensorShape({1}));
        }
        
        std::cout << "shared_name: " << shared_name << std::endl;
        std::cout << "container: " << container << std::endl;
        std::cout << "output_types size: " << output_types.size() << std::endl;
        std::cout << "output_shapes size: " << output_shapes.size() << std::endl;
        
        // Create the Iterator operation using the raw API
        tensorflow::NodeDef node_def;
        node_def.set_name("iterator");
        node_def.set_op("IteratorV2");
        
        auto attr_shared_name = node_def.mutable_attr()->insert({"shared_name", tensorflow::AttrValue()});
        attr_shared_name->second.set_s(shared_name);
        
        auto attr_container = node_def.mutable_attr()->insert({"container", tensorflow::AttrValue()});
        attr_container->second.set_s(container);
        
        auto attr_output_types = node_def.mutable_attr()->insert({"output_types", tensorflow::AttrValue()});
        for (const auto& dtype : output_types) {
            attr_output_types->second.mutable_list()->add_type(dtype);
        }
        
        auto attr_output_shapes = node_def.mutable_attr()->insert({"output_shapes", tensorflow::AttrValue()});
        for (const auto& shape : output_shapes) {
            auto* tensor_shape = attr_output_shapes->second.mutable_list()->add_shape();
            for (int i = 0; i < shape.dims(); ++i) {
                tensor_shape->add_dim()->set_size(shape.dim_size(i));
            }
        }
        
        tensorflow::Status status;
        auto iterator_op = root.AddNode(node_def, &status);
        
        if (!status.ok()) {
            std::cout << "Error creating iterator node: " << status.ToString() << std::endl;
            return -1;
        }
        
        std::cout << "Iterator operation created successfully" << std::endl;
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        status = session.Run({iterator_op}, &outputs);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }
        
        std::cout << "Session run successfully, outputs size: " << outputs.size() << std::endl;

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
