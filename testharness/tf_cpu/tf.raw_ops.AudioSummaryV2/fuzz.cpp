#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
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
            dtype = tensorflow::DT_STRING;
            break;
        case 2:
            dtype = tensorflow::DT_FLOAT;
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

void fillStringTensor(tensorflow::Tensor& tensor, const uint8_t* data,
                      size_t& offset, size_t total_size) {
    auto flat = tensor.flat<tensorflow::tstring>();
    const size_t num_elements = flat.size();
    
    for (size_t i = 0; i < num_elements; ++i) {
        if (offset < total_size) {
            uint8_t str_len = data[offset] % 20 + 1;
            offset++;
            
            std::string str;
            for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                str += static_cast<char>(data[offset] % 128);
                offset++;
            }
            flat(i) = tensorflow::tstring(str);
        } else {
            flat(i) = tensorflow::tstring("default");
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
        case tensorflow::DT_STRING:
            fillStringTensor(tensor, data, offset, total_size);
            break;
        default:
            fillTensorWithData<float>(tensor, data, offset, total_size);
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType tag_dtype = tensorflow::DT_STRING;
        uint8_t tag_rank = 0;
        std::vector<int64_t> tag_shape = {};
        tensorflow::TensorShape tag_tensor_shape(tag_shape);
        tensorflow::Tensor tag_tensor(tag_dtype, tag_tensor_shape);
        fillTensorWithDataByType(tag_tensor, tag_dtype, data, offset, size);

        tensorflow::DataType tensor_dtype = tensorflow::DT_FLOAT;
        uint8_t tensor_rank = parseRank(data[offset % size]);
        offset++;
        if (tensor_rank < 2) tensor_rank = 2;
        if (tensor_rank > 3) tensor_rank = 3;
        
        std::vector<int64_t> tensor_shape = parseShape(data, offset, size, tensor_rank);
        if (tensor_shape.size() >= 2) {
            tensor_shape[tensor_shape.size()-1] = std::min(tensor_shape[tensor_shape.size()-1], static_cast<int64_t>(2));
        }
        
        tensorflow::TensorShape tensor_tensor_shape(tensor_shape);
        tensorflow::Tensor tensor_tensor(tensor_dtype, tensor_tensor_shape);
        fillTensorWithDataByType(tensor_tensor, tensor_dtype, data, offset, size);
        
        auto flat = tensor_tensor.flat<float>();
        for (int i = 0; i < flat.size(); ++i) {
            flat(i) = std::max(-1.0f, std::min(1.0f, flat(i)));
        }

        tensorflow::DataType sample_rate_dtype = tensorflow::DT_FLOAT;
        uint8_t sample_rate_rank = 0;
        std::vector<int64_t> sample_rate_shape = {};
        tensorflow::TensorShape sample_rate_tensor_shape(sample_rate_shape);
        tensorflow::Tensor sample_rate_tensor(sample_rate_dtype, sample_rate_tensor_shape);
        fillTensorWithDataByType(sample_rate_tensor, sample_rate_dtype, data, offset, size);
        
        auto sample_rate_flat = sample_rate_tensor.flat<float>();
        sample_rate_flat(0) = std::max(1.0f, std::abs(sample_rate_flat(0)));

        int max_outputs = 1;
        if (offset < size) {
            max_outputs = std::max(1, static_cast<int>(data[offset] % 5 + 1));
            offset++;
        }

        auto tag_input = tensorflow::ops::Const(root, tag_tensor);
        auto tensor_input = tensorflow::ops::Const(root, tensor_tensor);
        auto sample_rate_input = tensorflow::ops::Const(root, sample_rate_tensor);

        // Use raw_ops.AudioSummaryV2 instead of ops::AudioSummaryV2
        auto audio_summary = tensorflow::ops::internal::AudioSummaryV2(
            root, tag_input, tensor_input, sample_rate_input, max_outputs);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({audio_summary}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
