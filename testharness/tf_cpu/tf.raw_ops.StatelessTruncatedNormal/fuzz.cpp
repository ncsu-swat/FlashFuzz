#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/random_ops.h"
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
        std::cerr << "Error: " << message << std::endl;
    }
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 2:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 3:
            dtype = tensorflow::DT_DOUBLE;
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
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BFLOAT16:
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
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
        tensorflow::DataType output_dtype = parseDataType(data[offset++]);
        
        uint8_t shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> shape_dims = parseShape(data, offset, size, shape_rank);
        
        tensorflow::TensorShape shape_tensor_shape({static_cast<int64_t>(shape_dims.size())});
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT32, shape_tensor_shape);
        auto shape_flat = shape_tensor.flat<int32_t>();
        for (size_t i = 0; i < shape_dims.size(); ++i) {
            shape_flat(i) = static_cast<int32_t>(shape_dims[i]);
        }
        
        tensorflow::TensorShape seed_tensor_shape({2});
        tensorflow::Tensor seed_tensor(tensorflow::DT_INT32, seed_tensor_shape);
        fillTensorWithDataByType(seed_tensor, tensorflow::DT_INT32, data, offset, size);
        
        auto shape_input = tensorflow::ops::Const(root, shape_tensor);
        auto seed_input = tensorflow::ops::Const(root, seed_tensor);
        
        std::cout << "Shape tensor: ";
        for (int i = 0; i < shape_tensor.NumElements(); ++i) {
            std::cout << shape_flat(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Seed tensor: ";
        auto seed_flat = seed_tensor.flat<int32_t>();
        for (int i = 0; i < seed_tensor.NumElements(); ++i) {
            std::cout << seed_flat(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Output dtype: " << output_dtype << std::endl;
        
        // Use the raw_ops API to call StatelessTruncatedNormal
        auto result = tensorflow::ops::StatelessRandomNormal(
            root, shape_input, seed_input,
            tensorflow::ops::StatelessRandomNormal::Dtype(output_dtype));
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({result}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }
        
        if (!outputs.empty()) {
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
            std::cout << "Output tensor dtype: " << outputs[0].dtype() << std::endl;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
