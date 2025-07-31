#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10
#define MIN_NUM_SHAPES 2
#define MAX_NUM_SHAPES 5

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << "Error: " << message << std::endl;
    }
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
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
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        default:
            return;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        if (offset >= size) return 0;
        uint8_t concat_dim_val = data[offset++];
        concat_dim_val = concat_dim_val % MAX_RANK;
        
        tensorflow::Tensor concat_dim_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        concat_dim_tensor.scalar<int32_t>()() = static_cast<int32_t>(concat_dim_val);
        auto concat_dim_op = tensorflow::ops::Const(root, concat_dim_tensor);

        if (offset >= size) return 0;
        uint8_t num_shapes_byte = data[offset++];
        int num_shapes = (num_shapes_byte % (MAX_NUM_SHAPES - MIN_NUM_SHAPES + 1)) + MIN_NUM_SHAPES;

        if (offset >= size) return 0;
        tensorflow::DataType shape_dtype = parseDataType(data[offset++]);

        std::vector<tensorflow::Output> shape_ops;
        
        for (int i = 0; i < num_shapes; ++i) {
            if (offset >= size) return 0;
            uint8_t rank = parseRank(data[offset++]);
            
            std::vector<int64_t> shape_dims = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape_dims) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor shape_tensor(shape_dtype, tensor_shape);
            fillTensorWithDataByType(shape_tensor, shape_dtype, data, offset, size);
            
            auto shape_op = tensorflow::ops::Const(root, shape_tensor);
            shape_ops.push_back(shape_op);
        }

        std::vector<tensorflow::Input> shape_inputs;
        for (const auto& shape_op : shape_ops) {
            shape_inputs.push_back(shape_op);
        }
        
        // Use raw_ops.ConcatOffset
        auto concat_offset_op = tensorflow::ops::_ConcatOffsetV2(root, concat_dim_op, shape_inputs);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({concat_offset_op}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}