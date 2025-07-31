#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/dataset_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
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

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
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
                        str += static_cast<char>(data[offset] % 128);
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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::Tensor multi_device_iterator_tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
        
        if (offset >= size) return 0;
        int32_t shard_num_value = 0;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&shard_num_value, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            shard_num_value = std::abs(shard_num_value) % 10;
        }
        tensorflow::Tensor shard_num_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        shard_num_tensor.scalar<int32_t>()() = shard_num_value;
        
        if (offset >= size) return 0;
        int64_t incarnation_id_value = 0;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&incarnation_id_value, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            incarnation_id_value = std::abs(incarnation_id_value) % 1000;
        }
        tensorflow::Tensor incarnation_id_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        incarnation_id_tensor.scalar<int64_t>()() = incarnation_id_value;

        if (offset >= size) return 0;
        uint8_t num_outputs = (data[offset] % 5) + 1;
        offset++;

        std::vector<tensorflow::DataType> output_types;
        std::vector<tensorflow::TensorShape> output_shapes;
        
        for (uint8_t i = 0; i < num_outputs; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType dtype = parseDataType(data[offset]);
            offset++;
            output_types.push_back(dtype);
            
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

        auto multi_device_iterator = tensorflow::ops::Placeholder(root, tensorflow::DT_RESOURCE);
        auto shard_num = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto incarnation_id = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);

        // Create the operation using raw_ops
        auto scope = root.WithOpName("MultiDeviceIteratorGetNextFromShard");
        tensorflow::NodeDef node_def;
        node_def.set_name(scope.GetUniqueNameForOp("MultiDeviceIteratorGetNextFromShard"));
        node_def.set_op("MultiDeviceIteratorGetNextFromShard");
        
        // Add inputs
        node_def.add_input(multi_device_iterator.node()->name());
        node_def.add_input(shard_num.node()->name());
        node_def.add_input(incarnation_id.node()->name());
        
        // Add attributes
        tensorflow::AttrValue output_types_attr;
        for (const auto& dtype : output_types) {
            output_types_attr.mutable_list()->add_type(dtype);
        }
        (*node_def.mutable_attr())["output_types"] = output_types_attr;
        
        tensorflow::AttrValue output_shapes_attr;
        for (const auto& shape : output_shapes) {
            tensorflow::TensorShapeProto shape_proto;
            shape.AsProto(&shape_proto);
            *output_shapes_attr.mutable_list()->add_shape() = shape_proto;
        }
        (*node_def.mutable_attr())["output_shapes"] = output_shapes_attr;
        
        // Create the operation
        tensorflow::Status status;
        auto op = root.AddNode(node_def, &status);
        if (!status.ok()) {
            return -1;
        }
        
        std::vector<tensorflow::Output> result;
        for (int i = 0; i < output_types.size(); ++i) {
            result.push_back(tensorflow::Output(op, i));
        }

        tensorflow::ClientSession session(root);
        
        std::vector<std::pair<tensorflow::Output, tensorflow::Input::Initializer>> feed_dict = {
            {multi_device_iterator, multi_device_iterator_tensor},
            {shard_num, shard_num_tensor},
            {incarnation_id, incarnation_id_tensor}
        };

        std::vector<tensorflow::Tensor> outputs;
        std::vector<tensorflow::Output> output_tensors;
        for (const auto& output : result) {
            output_tensors.push_back(output);
        }

        status = session.Run(feed_dict, output_tensors, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}