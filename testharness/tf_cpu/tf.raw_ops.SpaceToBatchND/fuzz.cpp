#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

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
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 12:
            dtype = tensorflow::DT_HALF;
            break;
        case 13:
            dtype = tensorflow::DT_UINT32;
            break;
        case 14:
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
        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        
        uint8_t input_rank = parseRank(data[offset++]);
        if (input_rank < 3) input_rank = 3;
        
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        if (input_shape.size() < 3) {
            return 0;
        }
        
        tensorflow::TensorShape input_tensor_shape;
        for (auto dim : input_shape) {
            input_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_tensor(input_dtype, input_tensor_shape);
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        
        int spatial_dims = input_rank - 2;
        if (spatial_dims <= 0) spatial_dims = 1;
        
        std::vector<int32_t> block_shape_data;
        for (int i = 0; i < spatial_dims; ++i) {
            if (offset < size) {
                int32_t block_val = (data[offset++] % 3) + 1;
                block_shape_data.push_back(block_val);
            } else {
                block_shape_data.push_back(2);
            }
        }
        
        tensorflow::TensorShape block_shape_tensor_shape;
        block_shape_tensor_shape.AddDim(spatial_dims);
        tensorflow::Tensor block_shape_tensor(tensorflow::DT_INT32, block_shape_tensor_shape);
        auto block_shape_flat = block_shape_tensor.flat<int32_t>();
        for (int i = 0; i < spatial_dims; ++i) {
            block_shape_flat(i) = block_shape_data[i];
        }
        
        std::vector<int32_t> paddings_data;
        for (int i = 0; i < spatial_dims; ++i) {
            if (offset + 1 < size) {
                int32_t pad_start = data[offset++] % 3;
                int32_t pad_end = data[offset++] % 3;
                paddings_data.push_back(pad_start);
                paddings_data.push_back(pad_end);
            } else {
                paddings_data.push_back(0);
                paddings_data.push_back(0);
            }
        }
        
        tensorflow::TensorShape paddings_tensor_shape;
        paddings_tensor_shape.AddDim(spatial_dims);
        paddings_tensor_shape.AddDim(2);
        tensorflow::Tensor paddings_tensor(tensorflow::DT_INT32, paddings_tensor_shape);
        auto paddings_flat = paddings_tensor.flat<int32_t>();
        for (int i = 0; i < spatial_dims * 2; ++i) {
            paddings_flat(i) = paddings_data[i];
        }
        
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto block_shape_op = tensorflow::ops::Const(root, block_shape_tensor);
        auto paddings_op = tensorflow::ops::Const(root, paddings_tensor);
        
        auto space_to_batch_nd = tensorflow::ops::SpaceToBatchND(root, input_op, block_shape_op, paddings_op);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({space_to_batch_nd}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
