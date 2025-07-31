#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
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
            }
            break;
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        if (offset >= size) return 0;
        int64_t seed_val;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&seed_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        } else {
            seed_val = 42;
        }
        
        if (offset >= size) return 0;
        int64_t seed2_val;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&seed2_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        } else {
            seed2_val = 24;
        }

        tensorflow::Tensor seed_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        seed_tensor.scalar<int64_t>()() = seed_val;
        
        tensorflow::Tensor seed2_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        seed2_tensor.scalar<int64_t>()() = seed2_val;

        auto seed_op = tensorflow::ops::Const(root, seed_tensor);
        auto seed2_op = tensorflow::ops::Const(root, seed2_tensor);

        tensorflow::Tensor dummy_resource(tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
        auto seed_generator_op = tensorflow::ops::Const(root, dummy_resource);

        if (offset >= size) return 0;
        uint8_t num_output_types = (data[offset] % 3) + 1;
        offset++;

        std::vector<tensorflow::DataType> output_types;
        std::vector<tensorflow::PartialTensorShape> output_shapes;

        for (uint8_t i = 0; i < num_output_types; ++i) {
            if (offset >= size) break;
            tensorflow::DataType dtype = parseDataType(data[offset]);
            offset++;
            output_types.push_back(dtype);

            if (offset >= size) break;
            uint8_t rank = parseRank(data[offset]);
            offset++;
            
            std::vector<int64_t> shape_dims = parseShape(data, offset, size, rank);
            tensorflow::PartialTensorShape shape(shape_dims);
            output_shapes.push_back(shape);
        }

        if (output_types.empty()) {
            output_types.push_back(tensorflow::DT_INT64);
            output_shapes.push_back(tensorflow::PartialTensorShape({}));
        }

        bool rerandomize = false;
        if (offset < size) {
            rerandomize = (data[offset] % 2) == 1;
            offset++;
        }

        std::string metadata = "";
        if (offset < size) {
            uint8_t metadata_len = data[offset] % 10;
            offset++;
            for (uint8_t i = 0; i < metadata_len && offset < size; ++i) {
                metadata += static_cast<char>(data[offset]);
                offset++;
            }
        }

        std::cout << "Creating RandomDatasetV2 with:" << std::endl;
        std::cout << "  seed: " << seed_val << std::endl;
        std::cout << "  seed2: " << seed2_val << std::endl;
        std::cout << "  output_types size: " << output_types.size() << std::endl;
        std::cout << "  output_shapes size: " << output_shapes.size() << std::endl;
        std::cout << "  rerandomize: " << rerandomize << std::endl;
        std::cout << "  metadata: " << metadata << std::endl;

        // Use raw op instead of the ops namespace
        tensorflow::NodeDef node_def;
        node_def.set_op("RandomDatasetV2");
        node_def.set_name("random_dataset");
        
        // Add inputs
        tensorflow::NodeDefBuilder builder(node_def.name(), node_def.op());
        builder.Input(seed_op.node()->name(), 0, tensorflow::DT_INT64)
               .Input(seed2_op.node()->name(), 0, tensorflow::DT_INT64)
               .Input(seed_generator_op.node()->name(), 0, tensorflow::DT_RESOURCE);
        
        // Add attributes
        builder.Attr("output_types", output_types);
        builder.Attr("output_shapes", output_shapes);
        builder.Attr("rerandomize_each_iteration", rerandomize);
        builder.Attr("metadata", metadata);
        
        // Finalize the NodeDef
        tensorflow::Status status = builder.Finalize(&node_def);
        if (!status.ok()) {
            std::cout << "Error creating NodeDef: " << status.ToString() << std::endl;
            return -1;
        }
        
        // Create the operation
        tensorflow::Operation random_dataset_op;
        status = root.AddNode(node_def, &random_dataset_op);
        if (!status.ok()) {
            std::cout << "Error adding node: " << status.ToString() << std::endl;
            return -1;
        }
        
        tensorflow::Output random_dataset(random_dataset_op, 0);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({random_dataset}, &outputs);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

        std::cout << "RandomDatasetV2 operation completed successfully" << std::endl;

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}