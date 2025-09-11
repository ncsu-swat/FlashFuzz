#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/dataset_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include <cstring>
#include <vector>
#include <iostream>

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
    switch (selector % 11) {  
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
            dtype = tensorflow::DT_UINT16;
            break;
        case 9:
            dtype = tensorflow::DT_UINT32;
            break;
        case 10:
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
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        uint8_t input_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        tensorflow::TensorShape input_tensor_shape;
        for (int64_t dim : input_shape) {
            input_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_dataset_tensor(tensorflow::DT_VARIANT, tensorflow::TensorShape({}));
        
        if (offset >= size) return 0;
        
        int64_t cycle_length_val = 1;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&cycle_length_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            cycle_length_val = std::abs(cycle_length_val) % 10 + 1;
        }
        
        int64_t block_length_val = 1;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&block_length_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            block_length_val = std::abs(block_length_val) % 10 + 1;
        }
        
        bool sloppy_val = false;
        if (offset < size) {
            sloppy_val = data[offset++] % 2;
        }
        
        int64_t buffer_output_elements_val = 1;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&buffer_output_elements_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            buffer_output_elements_val = std::abs(buffer_output_elements_val) % 100 + 1;
        }
        
        int64_t prefetch_input_elements_val = 1;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&prefetch_input_elements_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            prefetch_input_elements_val = std::abs(prefetch_input_elements_val) % 100 + 1;
        }

        auto input_dataset = tensorflow::ops::Const(root, input_dataset_tensor);
        auto cycle_length = tensorflow::ops::Const(root, cycle_length_val);
        auto block_length = tensorflow::ops::Const(root, block_length_val);
        auto sloppy = tensorflow::ops::Const(root, sloppy_val);
        auto buffer_output_elements = tensorflow::ops::Const(root, buffer_output_elements_val);
        auto prefetch_input_elements = tensorflow::ops::Const(root, prefetch_input_elements_val);

        std::vector<tensorflow::DataType> output_types = {input_dtype};
        std::vector<tensorflow::PartialTensorShape> output_shapes = {tensorflow::PartialTensorShape(input_shape)};

        tensorflow::Node* node;
        tensorflow::NodeBuilder builder("experimental_parallel_interleave", "ExperimentalParallelInterleaveDataset");
        builder.Input(input_dataset.node())
               .Input(std::vector<tensorflow::NodeBuilder::NodeOut>{})
               .Input(cycle_length.node())
               .Input(block_length.node())
               .Input(sloppy.node())
               .Input(buffer_output_elements.node())
               .Input(prefetch_input_elements.node())
               .Attr("f", tensorflow::NameAttrList())
               .Attr("output_types", output_types)
               .Attr("output_shapes", output_shapes);
        
        tensorflow::Status build_status = builder.Finalize(root.graph(), &node);
        if (!build_status.ok()) {
            return -1;
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({tensorflow::Output(node)}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
