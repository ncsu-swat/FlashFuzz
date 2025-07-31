#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/io_ops.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>

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
    switch (selector % 1) {
        case 0:
            dtype = tensorflow::DT_STRING;
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
            size_t str_len = std::min(static_cast<size_t>(data[offset] % 100 + 1), total_size - offset - 1);
            offset++;
            
            if (offset + str_len <= total_size) {
                std::string str(reinterpret_cast<const char*>(data + offset), str_len);
                flat(i) = tensorflow::tstring(str);
                offset += str_len;
            } else {
                flat(i) = tensorflow::tstring("default");
                offset = total_size;
            }
        } else {
            flat(i) = tensorflow::tstring("default");
        }
    }
}

std::string createTempFile(const std::string& content, const std::string& suffix) {
    std::string temp_dir = "/tmp";
    std::string filename = temp_dir + "/vocab_" + suffix + "_" + std::to_string(rand()) + ".txt";
    
    std::ofstream file(filename);
    if (file.is_open()) {
        file << content;
        file.close();
    }
    
    return filename;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType new_vocab_dtype = parseDataType(data[offset++]);
        uint8_t new_vocab_rank = parseRank(data[offset++]);
        std::vector<int64_t> new_vocab_shape = parseShape(data, offset, size, new_vocab_rank);
        
        tensorflow::DataType old_vocab_dtype = parseDataType(data[offset++]);
        uint8_t old_vocab_rank = parseRank(data[offset++]);
        std::vector<int64_t> old_vocab_shape = parseShape(data, offset, size, old_vocab_rank);

        if (offset >= size) return 0;

        int32_t new_vocab_offset_val = 0;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&new_vocab_offset_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            new_vocab_offset_val = std::abs(new_vocab_offset_val) % 10;
        }

        int32_t num_new_vocab_val = 3;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&num_new_vocab_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            num_new_vocab_val = std::abs(num_new_vocab_val) % 10 + 1;
        }

        int32_t old_vocab_size_val = -1;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&old_vocab_size_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            if (old_vocab_size_val >= 0) {
                old_vocab_size_val = std::abs(old_vocab_size_val) % 10 + 1;
            } else {
                old_vocab_size_val = -1;
            }
        }

        std::string new_vocab_content = "word0\nword1\nword2\nword3\nword4\n";
        std::string old_vocab_content = "word1\nword0\nword3\n";
        
        std::string new_vocab_file = createTempFile(new_vocab_content, "new");
        std::string old_vocab_file = createTempFile(old_vocab_content, "old");

        tensorflow::Tensor new_vocab_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        new_vocab_tensor.scalar<tensorflow::tstring>()() = tensorflow::tstring(new_vocab_file);

        tensorflow::Tensor old_vocab_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        old_vocab_tensor.scalar<tensorflow::tstring>()() = tensorflow::tstring(old_vocab_file);

        auto new_vocab_input = tensorflow::ops::Const(root, new_vocab_tensor);
        auto old_vocab_input = tensorflow::ops::Const(root, old_vocab_tensor);

        // Use raw_ops namespace for GenerateVocabRemapping
        auto generate_vocab_remapping = tensorflow::ops::internal::GenerateVocabRemapping(
            root, 
            new_vocab_input, 
            old_vocab_input,
            tensorflow::ops::internal::GenerateVocabRemapping::Attrs()
                .new_vocab_offset_(new_vocab_offset_val)
                .num_new_vocab_(num_new_vocab_val)
                .old_vocab_size_(old_vocab_size_val)
        );

        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({generate_vocab_remapping.remapping, generate_vocab_remapping.num_present}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

        std::remove(new_vocab_file.c_str());
        std::remove(old_vocab_file.c_str());

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}