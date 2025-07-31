#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/lookup_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

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
            dtype = tensorflow::DT_RESOURCE;
            break;
        case 1:
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
            size_t str_len = std::min(static_cast<size_t>(32), total_size - offset);
            if (str_len > 0) {
                std::string str(reinterpret_cast<const char*>(data + offset), str_len);
                flat(i) = tensorflow::tstring(str);
                offset += str_len;
            } else {
                flat(i) = tensorflow::tstring("");
            }
        } else {
            flat(i) = tensorflow::tstring("");
        }
    }
}

void fillTensorWithDataByType(tensorflow::Tensor& tensor,
                              tensorflow::DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
    switch (dtype) {
        case tensorflow::DT_STRING:
            fillStringTensor(tensor, data, offset, total_size);
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
        tensorflow::DataType table_dtype = parseDataType(data[offset++]);
        uint8_t table_rank = parseRank(data[offset++]);
        std::vector<int64_t> table_shape = parseShape(data, offset, size, table_rank);
        
        tensorflow::Tensor table_handle_tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape(table_shape));
        
        tensorflow::DataType filename_dtype = tensorflow::DT_STRING;
        uint8_t filename_rank = parseRank(data[offset++]);
        std::vector<int64_t> filename_shape = parseShape(data, offset, size, filename_rank);
        
        tensorflow::Tensor filename_tensor(filename_dtype, tensorflow::TensorShape(filename_shape));
        fillTensorWithDataByType(filename_tensor, filename_dtype, data, offset, size);
        
        int32_t key_index = -2;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&key_index, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            key_index = std::max<int32_t>(-2, key_index % 10);
        }
        
        int32_t value_index = -2;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&value_index, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            value_index = std::max<int32_t>(-2, value_index % 10);
        }
        
        int64_t vocab_size = -1;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&vocab_size, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            vocab_size = std::max<int64_t>(-1, vocab_size % 1000);
        }
        
        std::string delimiter = "\t";
        if (offset < size) {
            size_t delim_len = std::min(static_cast<size_t>(4), size - offset);
            if (delim_len > 0) {
                delimiter = std::string(reinterpret_cast<const char*>(data + offset), delim_len);
                offset += delim_len;
            }
        }
        
        int64_t table_offset = 0;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&table_offset, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            table_offset = std::max<int64_t>(0, table_offset % 100);
        }

        std::string temp_filename = "/tmp/test_vocab_" + std::to_string(reinterpret_cast<uintptr_t>(data)) + ".txt";
        std::ofstream temp_file(temp_filename);
        if (temp_file.is_open()) {
            temp_file << "key1\tvalue1\n";
            temp_file << "key2\tvalue2\n";
            temp_file << "key3\tvalue3\n";
            temp_file.close();
        }

        auto filename_input = tensorflow::ops::Const(root, temp_filename);
        auto table_handle_input = tensorflow::ops::Const(root, table_handle_tensor);

        auto init_op = tensorflow::ops::InitializeTableFromTextFile(
            root,
            table_handle_input,
            filename_input,
            tensorflow::ops::InitializeTableFromTextFile::KeyIndex(key_index)
                .ValueIndex(value_index)
                .VocabSize(vocab_size)
                .Delimiter(delimiter)
        );

        tensorflow::ClientSession session(root);
        
        std::cout << "Table handle shape: ";
        for (auto dim : table_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Filename shape: ";
        for (auto dim : filename_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Key index: " << key_index << std::endl;
        std::cout << "Value index: " << value_index << std::endl;
        std::cout << "Vocab size: " << vocab_size << std::endl;
        std::cout << "Delimiter: " << delimiter << std::endl;
        std::cout << "Offset: " << table_offset << std::endl;

        tensorflow::Status status = session.Run({init_op}, {});
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            std::remove(temp_filename.c_str());
            return -1;
        }

        std::remove(temp_filename.c_str());

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}