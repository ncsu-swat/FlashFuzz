#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
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
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
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
        tensorflow::DataType inputs_dtype = parseDataType(data[offset++]);
        
        uint8_t inputs_rank = 3;
        std::vector<int64_t> inputs_shape = {3, 2, 4};
        
        if (offset + 3 * sizeof(int64_t) <= size) {
            int64_t max_time, batch_size, num_classes;
            std::memcpy(&max_time, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            std::memcpy(&batch_size, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            std::memcpy(&num_classes, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            max_time = 1 + (std::abs(max_time) % 5);
            batch_size = 1 + (std::abs(batch_size) % 3);
            num_classes = 2 + (std::abs(num_classes) % 8);
            
            inputs_shape = {max_time, batch_size, num_classes};
        }

        tensorflow::TensorShape inputs_tensor_shape(inputs_shape);
        tensorflow::Tensor inputs_tensor(inputs_dtype, inputs_tensor_shape);
        fillTensorWithDataByType(inputs_tensor, inputs_dtype, data, offset, size);

        std::vector<int64_t> seq_len_shape = {inputs_shape[1]};
        tensorflow::TensorShape seq_len_tensor_shape(seq_len_shape);
        tensorflow::Tensor seq_len_tensor(tensorflow::DT_INT32, seq_len_tensor_shape);
        
        auto seq_len_flat = seq_len_tensor.flat<int32_t>();
        for (int i = 0; i < seq_len_flat.size(); ++i) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t seq_len;
                std::memcpy(&seq_len, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                seq_len = 1 + (std::abs(seq_len) % static_cast<int32_t>(inputs_shape[0]));
                seq_len_flat(i) = seq_len;
            } else {
                seq_len_flat(i) = 1;
            }
        }

        int beam_width = 2;
        int top_paths = 1;
        bool merge_repeated = true;
        
        if (offset + sizeof(int) <= size) {
            std::memcpy(&beam_width, data + offset, sizeof(int));
            offset += sizeof(int);
            beam_width = 1 + (std::abs(beam_width) % 5);
        }
        
        if (offset + sizeof(int) <= size) {
            std::memcpy(&top_paths, data + offset, sizeof(int));
            offset += sizeof(int);
            top_paths = 1 + (std::abs(top_paths) % beam_width);
        }
        
        if (offset < size) {
            merge_repeated = (data[offset] % 2) == 1;
            offset++;
        }

        auto inputs_placeholder = tensorflow::ops::Placeholder(root, inputs_dtype);
        auto seq_len_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);

        // Use raw_ops.CTCBeamSearchDecoder instead of ops::CTCBeamSearchDecoder
        std::vector<tensorflow::Output> decoded_indices;
        std::vector<tensorflow::Output> decoded_values;
        std::vector<tensorflow::Output> decoded_shape;
        tensorflow::Output log_probability;

        tensorflow::ops::CTCBeamSearchDecoderV2 ctc_decoder(
            root,
            inputs_placeholder,
            seq_len_placeholder,
            beam_width,
            top_paths,
            merge_repeated
        );

        decoded_indices = ctc_decoder.decoded_indices;
        decoded_values = ctc_decoder.decoded_values;
        decoded_shape = ctc_decoder.decoded_shape;
        log_probability = ctc_decoder.log_probability;

        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{inputs_placeholder, inputs_tensor}, {seq_len_placeholder, seq_len_tensor}},
            {decoded_indices[0], decoded_values[0], decoded_shape[0], log_probability},
            &outputs
        );
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
