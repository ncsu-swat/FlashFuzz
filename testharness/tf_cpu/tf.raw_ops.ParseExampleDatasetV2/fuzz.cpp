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
#include <string>
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
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
            break;
        case 2:
            dtype = tensorflow::DT_STRING;
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
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
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
                    flat(i) = tensorflow::tstring(str);
                } else {
                    flat(i) = tensorflow::tstring("");
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
        // Create a simple dataset as input
        auto range_dataset = tensorflow::ops::RangeDataset(
            root,
            tensorflow::Input::Initializer(0),
            tensorflow::Input::Initializer(10),
            tensorflow::Input::Initializer(1),
            {tensorflow::DT_INT64}
        );
        
        tensorflow::Tensor num_parallel_calls(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        num_parallel_calls.scalar<int64_t>()() = 1;

        std::vector<tensorflow::Tensor> dense_defaults;
        uint8_t num_dense_defaults = data[offset++] % 3 + 1;
        for (uint8_t i = 0; i < num_dense_defaults; ++i) {
            tensorflow::DataType dtype = parseDataType(data[offset++]);
            uint8_t rank = parseRank(data[offset++]);
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor default_tensor(dtype, tensor_shape);
            fillTensorWithDataByType(default_tensor, dtype, data, offset, size);
            dense_defaults.push_back(default_tensor);
        }

        std::vector<std::string> sparse_keys;
        uint8_t num_sparse_keys = data[offset++] % 3 + 1;
        for (uint8_t i = 0; i < num_sparse_keys; ++i) {
            if (offset < size) {
                uint8_t key_len = data[offset++] % 10 + 1;
                std::string key = "sparse_key_" + std::to_string(i);
                sparse_keys.push_back(key);
            }
        }

        std::vector<std::string> dense_keys;
        uint8_t num_dense_keys = data[offset++] % 3 + 1;
        for (uint8_t i = 0; i < num_dense_keys; ++i) {
            std::string key = "dense_key_" + std::to_string(i);
            dense_keys.push_back(key);
        }

        std::vector<tensorflow::DataType> sparse_types;
        for (uint8_t i = 0; i < num_sparse_keys; ++i) {
            if (offset < size) {
                tensorflow::DataType dtype = parseDataType(data[offset++]);
                sparse_types.push_back(dtype);
            } else {
                sparse_types.push_back(tensorflow::DT_FLOAT);
            }
        }

        std::vector<tensorflow::PartialTensorShape> dense_shapes;
        for (uint8_t i = 0; i < num_dense_keys; ++i) {
            if (offset < size) {
                uint8_t rank = parseRank(data[offset++]);
                std::vector<int64_t> shape = parseShape(data, offset, size, rank);
                tensorflow::PartialTensorShape tensor_shape(shape);
                dense_shapes.push_back(tensor_shape);
            } else {
                dense_shapes.push_back(tensorflow::PartialTensorShape({}));
            }
        }

        std::vector<tensorflow::DataType> output_types;
        uint8_t num_output_types = data[offset++] % 5 + 1;
        for (uint8_t i = 0; i < num_output_types; ++i) {
            if (offset < size) {
                tensorflow::DataType dtype = parseDataType(data[offset++]);
                output_types.push_back(dtype);
            } else {
                output_types.push_back(tensorflow::DT_FLOAT);
            }
        }

        std::vector<tensorflow::PartialTensorShape> output_shapes;
        for (uint8_t i = 0; i < num_output_types; ++i) {
            if (offset < size) {
                uint8_t rank = parseRank(data[offset++]);
                std::vector<int64_t> shape = parseShape(data, offset, size, rank);
                tensorflow::PartialTensorShape tensor_shape(shape);
                output_shapes.push_back(tensor_shape);
            } else {
                output_shapes.push_back(tensorflow::PartialTensorShape({}));
            }
        }

        bool sloppy = false;
        std::vector<std::string> ragged_keys;
        std::vector<tensorflow::DataType> ragged_value_types;
        std::vector<tensorflow::DataType> ragged_split_types;

        // Use raw_ops approach with NodeDef
        tensorflow::NodeDef node_def;
        node_def.set_op("ParseExampleDatasetV2");
        node_def.set_name("parse_example_dataset_v2");
        
        // Add inputs to NodeDef
        tensorflow::AddNodeInput("range_dataset", &node_def);
        tensorflow::AddNodeInput("num_parallel_calls", &node_def);
        
        // Add attributes to NodeDef
        tensorflow::AttrValue sparse_keys_attr;
        for (const auto& key : sparse_keys) {
            sparse_keys_attr.mutable_list()->add_s(key);
        }
        (*node_def.mutable_attr())["sparse_keys"] = sparse_keys_attr;
        
        tensorflow::AttrValue dense_keys_attr;
        for (const auto& key : dense_keys) {
            dense_keys_attr.mutable_list()->add_s(key);
        }
        (*node_def.mutable_attr())["dense_keys"] = dense_keys_attr;
        
        tensorflow::AttrValue sparse_types_attr;
        for (const auto& type : sparse_types) {
            sparse_types_attr.mutable_list()->add_type(type);
        }
        (*node_def.mutable_attr())["sparse_types"] = sparse_types_attr;
        
        tensorflow::AttrValue dense_shapes_attr;
        for (const auto& shape : dense_shapes) {
            tensorflow::TensorShapeProto shape_proto;
            shape.AsProto(&shape_proto);
            *dense_shapes_attr.mutable_list()->add_shape() = shape_proto;
        }
        (*node_def.mutable_attr())["dense_shapes"] = dense_shapes_attr;
        
        tensorflow::AttrValue output_types_attr;
        for (const auto& type : output_types) {
            output_types_attr.mutable_list()->add_type(type);
        }
        (*node_def.mutable_attr())["output_types"] = output_types_attr;
        
        tensorflow::AttrValue output_shapes_attr;
        for (const auto& shape : output_shapes) {
            tensorflow::TensorShapeProto shape_proto;
            shape.AsProto(&shape_proto);
            *output_shapes_attr.mutable_list()->add_shape() = shape_proto;
        }
        (*node_def.mutable_attr())["output_shapes"] = output_shapes_attr;
        
        tensorflow::AttrValue sloppy_attr;
        sloppy_attr.set_b(sloppy);
        (*node_def.mutable_attr())["sloppy"] = sloppy_attr;
        
        tensorflow::AttrValue ragged_keys_attr;
        for (const auto& key : ragged_keys) {
            ragged_keys_attr.mutable_list()->add_s(key);
        }
        (*node_def.mutable_attr())["ragged_keys"] = ragged_keys_attr;
        
        tensorflow::AttrValue ragged_value_types_attr;
        for (const auto& type : ragged_value_types) {
            ragged_value_types_attr.mutable_list()->add_type(type);
        }
        (*node_def.mutable_attr())["ragged_value_types"] = ragged_value_types_attr;
        
        tensorflow::AttrValue ragged_split_types_attr;
        for (const auto& type : ragged_split_types) {
            ragged_split_types_attr.mutable_list()->add_type(type);
        }
        (*node_def.mutable_attr())["ragged_split_types"] = ragged_split_types_attr;
        
        // Create operation using the NodeDef
        tensorflow::Status status;
        auto op = root.AddNode(node_def, &status);
        
        if (!status.ok()) {
            return -1;
        }

        // Run the session
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