#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/io_ops.h"
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

std::string parseString(const uint8_t* data, size_t& offset, size_t total_size, size_t max_length = 100) {
    if (offset >= total_size) {
        return "";
    }
    
    size_t length = std::min(max_length, total_size - offset);
    if (length > 0) {
        uint8_t len_byte = data[offset];
        offset++;
        length = std::min(static_cast<size_t>(len_byte % 50 + 1), total_size - offset);
    }
    
    std::string result;
    result.reserve(length);
    for (size_t i = 0; i < length && offset < total_size; ++i) {
        char c = static_cast<char>(data[offset]);
        if (c == 0) c = 'a';
        result += c;
        offset++;
    }
    
    if (result.empty()) {
        result = "/tmp/test*.txt";
    }
    
    return result;
}

int32_t parseInt32(const uint8_t* data, size_t& offset, size_t total_size) {
    if (offset + sizeof(int32_t) <= total_size) {
        int32_t value;
        std::memcpy(&value, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        return value;
    }
    return 301;
}

float parseFloat(const uint8_t* data, size_t& offset, size_t total_size) {
    if (offset + sizeof(float) <= total_size) {
        float value;
        std::memcpy(&value, data + offset, sizeof(float));
        offset += sizeof(float);
        if (std::isnan(value) || std::isinf(value)) {
            return 0.0f;
        }
        return value;
    }
    return 0.0f;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) {
        return 0;
    }
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        std::string file_pattern = parseString(data, offset, size);
        std::cout << "file_pattern: " << file_pattern << std::endl;
        
        int32_t file_random_seed = parseInt32(data, offset, size);
        if (file_random_seed < 0) file_random_seed = std::abs(file_random_seed);
        std::cout << "file_random_seed: " << file_random_seed << std::endl;
        
        float file_shuffle_shift_ratio = parseFloat(data, offset, size);
        if (file_shuffle_shift_ratio < 0.0f) file_shuffle_shift_ratio = 0.0f;
        if (file_shuffle_shift_ratio > 1.0f) file_shuffle_shift_ratio = 1.0f;
        std::cout << "file_shuffle_shift_ratio: " << file_shuffle_shift_ratio << std::endl;
        
        int32_t file_buffer_size = parseInt32(data, offset, size);
        if (file_buffer_size <= 0) file_buffer_size = 10000;
        std::cout << "file_buffer_size: " << file_buffer_size << std::endl;
        
        int32_t file_parallelism = parseInt32(data, offset, size);
        if (file_parallelism <= 0) file_parallelism = 16;
        std::cout << "file_parallelism: " << file_parallelism << std::endl;
        
        int32_t batch_size = parseInt32(data, offset, size);
        if (batch_size <= 0) batch_size = 32;
        std::cout << "batch_size: " << batch_size << std::endl;
        
        std::string compression_type = parseString(data, offset, size, 20);
        std::cout << "compression_type: " << compression_type << std::endl;

        auto record_input_attrs = tensorflow::ops::RecordInput::Attrs()
            .FileRandomSeed(file_random_seed)
            .FileShuffleShiftRatio(file_shuffle_shift_ratio)
            .FileBufferSize(file_buffer_size)
            .FileParallelism(file_parallelism)
            .BatchSize(batch_size)
            .CompressionType(compression_type);

        auto record_input = tensorflow::ops::RecordInput(root, file_pattern, record_input_attrs);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({record_input}, &outputs);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

        if (!outputs.empty()) {
            std::cout << "Output tensor shape: " << outputs[0].shape().DebugString() << std::endl;
            std::cout << "Output tensor dtype: " << tensorflow::DataTypeString(outputs[0].dtype()) << std::endl;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
