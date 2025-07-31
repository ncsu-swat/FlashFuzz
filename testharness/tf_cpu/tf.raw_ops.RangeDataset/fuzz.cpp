#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_shape.h"
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
        case tensorflow::DT_STRING:
            {
                auto flat = tensor.flat<tensorflow::tstring>();
                for (int i = 0; i < flat.size(); ++i) {
                    flat(i) = "test_string";
                }
            }
            break;
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        if (offset + 3 > size) return 0;
        
        int64_t start_val, stop_val, step_val;
        std::memcpy(&start_val, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        
        if (offset + sizeof(int64_t) > size) return 0;
        std::memcpy(&stop_val, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        
        if (offset + sizeof(int64_t) > size) return 0;
        std::memcpy(&step_val, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        
        start_val = start_val % 1000;
        stop_val = start_val + (std::abs(stop_val) % 100) + 1;
        step_val = (step_val == 0) ? 1 : (std::abs(step_val) % 10) + 1;
        
        tensorflow::Tensor start_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        start_tensor.scalar<int64_t>()() = start_val;
        
        tensorflow::Tensor stop_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        stop_tensor.scalar<int64_t>()() = stop_val;
        
        tensorflow::Tensor step_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        step_tensor.scalar<int64_t>()() = step_val;
        
        auto start_op = tensorflow::ops::Const(root, start_tensor);
        auto stop_op = tensorflow::ops::Const(root, stop_tensor);
        auto step_op = tensorflow::ops::Const(root, step_tensor);
        
        if (offset >= size) return 0;
        uint8_t num_output_types = (data[offset] % 3) + 1;
        offset++;
        
        std::vector<tensorflow::DataType> output_types;
        std::vector<tensorflow::PartialTensorShape> output_shapes;
        
        for (int i = 0; i < num_output_types; ++i) {
            if (offset >= size) break;
            tensorflow::DataType dtype = parseDataType(data[offset]);
            offset++;
            output_types.push_back(dtype);
            
            if (offset >= size) break;
            uint8_t rank = parseRank(data[offset]);
            offset++;
            
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            output_shapes.push_back(tensorflow::PartialTensorShape(shape));
        }
        
        if (output_types.empty()) {
            output_types.push_back(tensorflow::DT_INT64);
            output_shapes.push_back(tensorflow::PartialTensorShape({}));
        }
        
        // Create RangeDataset using raw ops
        tensorflow::NodeDef node_def;
        node_def.set_name("RangeDataset");
        node_def.set_op("RangeDataset");
        
        // Add inputs
        tensorflow::NameAttrList names;
        node_def.add_input(start_op.node()->name());
        node_def.add_input(stop_op.node()->name());
        node_def.add_input(step_op.node()->name());
        
        // Add attributes
        auto attr_output_types = node_def.mutable_attr();
        (*attr_output_types)["output_types"].mutable_list()->Reserve(output_types.size());
        for (const auto& dtype : output_types) {
            (*attr_output_types)["output_types"].mutable_list()->add_type(dtype);
        }
        
        tensorflow::AttrValue shapes_attr;
        for (const auto& shape : output_shapes) {
            tensorflow::TensorShapeProto shape_proto;
            shape.AsProto(&shape_proto);
            *shapes_attr.mutable_list()->add_shape() = shape_proto;
        }
        (*attr_output_types)["output_shapes"] = shapes_attr;
        
        // Set metadata attribute
        (*attr_output_types)["metadata"].set_s("");
        
        // Set replicate_on_split attribute
        (*attr_output_types)["replicate_on_split"].set_b(false);
        
        // Create the operation
        tensorflow::Status status;
        auto op = tensorflow::Operation(root.WithOpName("RangeDataset").WithDevice("/cpu:0"), 
                                       "RangeDataset", 
                                       {start_op, stop_op, step_op}, 
                                       {{"output_types", output_types}, 
                                        {"output_shapes", output_shapes},
                                        {"metadata", ""},
                                        {"replicate_on_split", false}});
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({op}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}