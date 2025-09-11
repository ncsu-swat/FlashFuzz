#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/lookup_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>

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
            dtype = tensorflow::DT_STRING;
            break;
        case 1:
            dtype = tensorflow::DT_STRING;
            break;
        default:
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
                flat(i) = str;
                offset += str_len;
            } else {
                flat(i) = "default";
                offset = total_size;
            }
        } else {
            flat(i) = "default";
        }
    }
}

std::string createTempFile(const uint8_t* data, size_t& offset, size_t total_size) {
    std::string temp_filename = "/tmp/test_vocab_" + std::to_string(rand()) + ".txt";
    std::ofstream file(temp_filename);
    
    if (offset < total_size) {
        size_t content_len = std::min(static_cast<size_t>(100), total_size - offset);
        
        for (size_t i = 0; i < content_len && offset < total_size; ++i) {
            char c = static_cast<char>(data[offset] % 94 + 33);
            if (c == '\n' || c == '\t' || (data[offset] % 10 == 0)) {
                file << '\n';
            } else {
                file << c;
            }
            offset++;
        }
    } else {
        file << "key1\tvalue1\n";
        file << "key2\tvalue2\n";
    }
    
    file.close();
    return temp_filename;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;
    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType table_dtype = parseDataType(data[offset++]);
        uint8_t table_rank = parseRank(data[offset++]);
        std::vector<int64_t> table_shape = parseShape(data, offset, size, table_rank);
        
        tensorflow::Tensor table_handle_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(table_shape));
        fillStringTensor(table_handle_tensor, data, offset, size);
        
        tensorflow::DataType filename_dtype = tensorflow::DT_STRING;
        uint8_t filename_rank = parseRank(data[offset++]);
        std::vector<int64_t> filename_shape = parseShape(data, offset, size, filename_rank);
        
        std::string temp_file = createTempFile(data, offset, size);
        tensorflow::Tensor filename_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(filename_shape));
        auto filename_flat = filename_tensor.flat<tensorflow::tstring>();
        for (int i = 0; i < filename_flat.size(); ++i) {
            filename_flat(i) = temp_file;
        }
        
        int key_index = -2;
        int value_index = -1;
        int vocab_size = -1;
        std::string delimiter = "\t";
        int offset_param = 0;
        
        if (offset < size) {
            key_index = static_cast<int>(data[offset++] % 5) - 2;
        }
        if (offset < size) {
            value_index = static_cast<int>(data[offset++] % 5) - 2;
        }
        if (offset < size) {
            vocab_size = static_cast<int>(data[offset++] % 100) - 1;
        }
        if (offset < size) {
            offset_param = static_cast<int>(data[offset++] % 10);
        }
        
        auto table_handle = tensorflow::ops::Placeholder(root, tensorflow::DT_STRING);
        auto filename_input = tensorflow::ops::Placeholder(root, tensorflow::DT_STRING);
        
        auto init_op = tensorflow::ops::InitializeTableFromTextFile(
            root,
            table_handle,
            filename_input,
            key_index,
            value_index,
            tensorflow::ops::InitializeTableFromTextFile::VocabSize(vocab_size)
                .Delimiter(delimiter)
                .Offset(offset_param)
        );
        
        tensorflow::ClientSession session(root);
        
        std::unordered_map<tensorflow::Output, tensorflow::Input::Initializer> feed_dict = {
            {table_handle, table_handle_tensor},
            {filename_input, filename_tensor}
        };
        
        std::vector<tensorflow::Operation> ops_to_run = {init_op.operation};
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(feed_dict, {}, ops_to_run, &outputs);
        
        std::remove(temp_file.c_str());
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
