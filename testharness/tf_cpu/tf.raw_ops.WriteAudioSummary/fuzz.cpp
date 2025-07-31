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
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 5) {
        case 0:
            dtype = tensorflow::DT_RESOURCE;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
            break;
        case 2:
            dtype = tensorflow::DT_STRING;
            break;
        case 3:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 4:
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
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_STRING:
            fillStringTensor(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_RESOURCE:
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
        tensorflow::Tensor writer_tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
        
        uint8_t step_rank = parseRank(data[offset++]);
        std::vector<int64_t> step_shape = parseShape(data, offset, size, step_rank);
        tensorflow::Tensor step_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(step_shape));
        fillTensorWithDataByType(step_tensor, tensorflow::DT_INT64, data, offset, size);
        
        uint8_t tag_rank = parseRank(data[offset++]);
        std::vector<int64_t> tag_shape = parseShape(data, offset, size, tag_rank);
        tensorflow::Tensor tag_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(tag_shape));
        fillTensorWithDataByType(tag_tensor, tensorflow::DT_STRING, data, offset, size);
        
        uint8_t tensor_rank = parseRank(data[offset++]);
        std::vector<int64_t> tensor_shape = parseShape(data, offset, size, tensor_rank);
        tensorflow::Tensor audio_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(tensor_shape));
        fillTensorWithDataByType(audio_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        uint8_t sample_rate_rank = parseRank(data[offset++]);
        std::vector<int64_t> sample_rate_shape = parseShape(data, offset, size, sample_rate_rank);
        tensorflow::Tensor sample_rate_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(sample_rate_shape));
        fillTensorWithDataByType(sample_rate_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        int max_outputs = 3;
        if (offset < size) {
            max_outputs = (data[offset] % 5) + 1;
            offset++;
        }

        auto writer_input = tensorflow::ops::Placeholder(root, tensorflow::DT_RESOURCE);
        auto step_input = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        auto tag_input = tensorflow::ops::Placeholder(root, tensorflow::DT_STRING);
        auto tensor_input = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto sample_rate_input = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);

        // Use raw_ops directly instead of summary_ops.h
        tensorflow::Output write_audio_summary = tensorflow::ops::internal::WriteAudioSummary(
            root,
            writer_input,
            step_input,
            tag_input,
            tensor_input,
            sample_rate_input,
            tensorflow::ops::internal::WriteAudioSummary::MaxOutputs(max_outputs)
        );

        tensorflow::ClientSession session(root);
        
        std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs = {
            {writer_input.node()->name(), writer_tensor},
            {step_input.node()->name(), step_tensor},
            {tag_input.node()->name(), tag_tensor},
            {tensor_input.node()->name(), audio_tensor},
            {sample_rate_input.node()->name(), sample_rate_tensor}
        };

        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(inputs, {}, {write_audio_summary.node()->name()}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}