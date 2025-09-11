#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10
#define MAX_NUM_TENSORS 5

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << "Error: " << message << std::endl;
    }
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype; 
    switch (selector % 15) {  
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
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 9:
            dtype = tensorflow::DT_UINT16;
            break;
        case 10:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 11:
            dtype = tensorflow::DT_HALF;
            break;
        case 12:
            dtype = tensorflow::DT_UINT32;
            break;
        case 13:
            dtype = tensorflow::DT_UINT64;
            break;
        case 14:
            dtype = tensorflow::DT_COMPLEX128;
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_tensors_byte = data[offset++];
        uint8_t num_tensors = (num_tensors_byte % MAX_NUM_TENSORS) + 1;
        
        std::vector<tensorflow::Input> inputs;
        std::vector<tensorflow::TensorShape> shapes;
        
        for (uint8_t i = 0; i < num_tensors; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            if (offset >= size) break;
            
            uint8_t rank = parseRank(data[offset++]);
            if (offset >= size) break;
            
            std::vector<int64_t> shape_dims = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape_dims) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor tensor(dtype, tensor_shape);
            
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            
            auto const_op = tensorflow::ops::Const(root, tensor);
            inputs.push_back(const_op);
            shapes.push_back(tensor_shape);
            
            std::cout << "Tensor " << i << " - Type: " << tensorflow::DataTypeString(dtype) 
                      << ", Shape: " << tensor_shape.DebugString() << std::endl;
        }
        
        if (inputs.empty()) {
            return 0;
        }
        
        std::vector<int> layouts;
        if (offset < size) {
            uint8_t num_layouts = data[offset++] % 10;
            for (uint8_t i = 0; i < num_layouts && offset < size; ++i) {
                int layout_val = static_cast<int>(data[offset++]) - 128;
                layouts.push_back(layout_val);
            }
        }
        
        int device_ordinal = -1;
        if (offset < size) {
            device_ordinal = static_cast<int>(data[offset++]) - 128;
        }
        
        std::cout << "Number of inputs: " << inputs.size() << std::endl;
        std::cout << "Number of shapes: " << shapes.size() << std::endl;
        std::cout << "Number of layouts: " << layouts.size() << std::endl;
        std::cout << "Device ordinal: " << device_ordinal << std::endl;
        
        // Create the InfeedEnqueueTuple operation using raw_ops
        tensorflow::NodeDef node_def;
        node_def.set_op("InfeedEnqueueTuple");
        node_def.set_name("InfeedEnqueueTuple");
        
        // Set attributes
        auto attr_map = node_def.mutable_attr();
        
        // Set dtypes attribute
        tensorflow::AttrValue dtypes_attr;
        for (size_t i = 0; i < inputs.size(); ++i) {
            dtypes_attr.mutable_list()->add_type(inputs[i].tensor().dtype());
        }
        (*attr_map)["dtypes"] = dtypes_attr;
        
        // Set shapes attribute
        tensorflow::AttrValue shapes_attr;
        for (const auto& shape : shapes) {
            tensorflow::TensorShapeProto* shape_proto = shapes_attr.mutable_list()->add_shape();
            shape.AsProto(shape_proto);
        }
        (*attr_map)["shapes"] = shapes_attr;
        
        // Set layouts attribute if provided
        if (!layouts.empty()) {
            tensorflow::AttrValue layouts_attr;
            for (int layout : layouts) {
                layouts_attr.mutable_list()->add_i(layout);
            }
            (*attr_map)["layouts"] = layouts_attr;
        }
        
        // Set device_ordinal attribute if provided
        if (device_ordinal >= 0) {
            tensorflow::AttrValue device_ordinal_attr;
            device_ordinal_attr.set_i(device_ordinal);
            (*attr_map)["device_ordinal"] = device_ordinal_attr;
        }
        
        // Create the operation
        tensorflow::Status status;
        auto op = root.AddNode(node_def, &status);
        
        if (!status.ok()) {
            std::cout << "Error creating InfeedEnqueueTuple node: " << status.ToString() << std::endl;
            return -1;
        }
        
        // Add inputs to the operation
        for (size_t i = 0; i < inputs.size(); ++i) {
            status = root.UpdateEdge(inputs[i], 0, op, i);
            if (!status.ok()) {
                std::cout << "Error connecting input " << i << ": " << status.ToString() << std::endl;
                return -1;
            }
        }
        
        tensorflow::ClientSession session(root);
        status = session.Run({op}, nullptr);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
