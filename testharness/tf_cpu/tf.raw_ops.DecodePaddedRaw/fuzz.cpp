#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <cstring>
#include <vector>
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

tensorflow::DataType parseOutputDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 10) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 2:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 3:
            dtype = tensorflow::DT_INT32;
            break;
        case 4:
            dtype = tensorflow::DT_UINT16;
            break;
        case 5:
            dtype = tensorflow::DT_UINT8;
            break;
        case 6:
            dtype = tensorflow::DT_INT16;
            break;
        case 7:
            dtype = tensorflow::DT_INT8;
            break;
        case 8:
            dtype = tensorflow::DT_INT64;
            break;
        case 9:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        default:
            dtype = tensorflow::DT_FLOAT;
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

void fillStringTensor(tensorflow::Tensor& tensor, const uint8_t* data, size_t& offset, size_t total_size) {
    auto flat = tensor.flat<tensorflow::tstring>();
    const size_t num_elements = flat.size();
    
    for (size_t i = 0; i < num_elements; ++i) {
        if (offset < total_size) {
            size_t string_length = std::min(static_cast<size_t>(32), total_size - offset);
            if (offset + 1 <= total_size) {
                string_length = std::min(static_cast<size_t>(data[offset] % 32 + 1), total_size - offset - 1);
                offset++;
            }
            
            std::string str_value;
            if (offset + string_length <= total_size) {
                str_value = std::string(reinterpret_cast<const char*>(data + offset), string_length);
                offset += string_length;
            } else {
                str_value = "test";
            }
            flat(i) = str_value;
        } else {
            flat(i) = "test";
        }
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t input_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        tensorflow::TensorShape input_tensor_shape;
        for (int64_t dim : input_shape) {
            input_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_bytes_tensor(tensorflow::DT_STRING, input_tensor_shape);
        fillStringTensor(input_bytes_tensor, data, offset, size);
        
        if (offset >= size) return 0;
        
        int32_t fixed_length_value = 4;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&fixed_length_value, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            fixed_length_value = std::abs(fixed_length_value) % 64 + 1;
        }
        
        tensorflow::Tensor fixed_length_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        fixed_length_tensor.scalar<int32_t>()() = fixed_length_value;
        
        if (offset >= size) return 0;
        
        tensorflow::DataType out_type = parseOutputDataType(data[offset++]);
        
        bool little_endian = true;
        if (offset < size) {
            little_endian = (data[offset++] % 2) == 0;
        }
        
        auto input_bytes = tensorflow::ops::Const(root, input_bytes_tensor);
        auto fixed_length = tensorflow::ops::Const(root, fixed_length_tensor);
        
        auto decode_op = tensorflow::ops::DecodePaddedRaw(
            root,
            input_bytes,
            fixed_length,
            out_type,
            tensorflow::ops::DecodePaddedRaw::LittleEndian(little_endian)
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({decode_op}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}