#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/parsing_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <vector>
#include <cstring>
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
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
            dtype = tensorflow::DT_INT32;
            break;
        case 3:
            dtype = tensorflow::DT_INT64;
            break;
        case 4:
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
            size_t str_len = std::min(static_cast<size_t>(data[offset] % 100), total_size - offset - 1);
            offset++;
            
            if (offset + str_len <= total_size) {
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
        case tensorflow::DT_STRING:
            fillStringTensor(tensor, data, offset, total_size);
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
        uint8_t records_rank = parseRank(data[offset++]);
        std::vector<int64_t> records_shape = parseShape(data, offset, size, records_rank);
        
        tensorflow::Tensor records_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(records_shape));
        fillStringTensor(records_tensor, data, offset, size);
        
        auto records_input = tensorflow::ops::Const(root, records_tensor);

        uint8_t num_defaults = (data[offset++] % 5) + 1;
        std::vector<tensorflow::Input> record_defaults;
        
        for (uint8_t i = 0; i < num_defaults; ++i) {
            if (offset >= size) break;
            
            tensorflow::DataType default_dtype = parseDataType(data[offset++]);
            uint8_t default_rank = parseRank(data[offset++]);
            std::vector<int64_t> default_shape = parseShape(data, offset, size, default_rank);
            
            tensorflow::Tensor default_tensor(default_dtype, tensorflow::TensorShape(default_shape));
            fillTensorWithDataByType(default_tensor, default_dtype, data, offset, size);
            
            record_defaults.push_back(tensorflow::ops::Const(root, default_tensor));
        }

        std::string field_delim = ",";
        if (offset < size) {
            char delim_char = static_cast<char>(data[offset++] % 128);
            if (delim_char >= 32 && delim_char <= 126) {
                field_delim = std::string(1, delim_char);
            }
        }

        bool use_quote_delim = true;
        if (offset < size) {
            use_quote_delim = (data[offset++] % 2) == 1;
        }

        std::string na_value = "";
        if (offset < size) {
            size_t na_len = std::min(static_cast<size_t>(data[offset] % 10), size - offset - 1);
            offset++;
            if (offset + na_len <= size) {
                na_value = std::string(reinterpret_cast<const char*>(data + offset), na_len);
                offset += na_len;
            }
        }

        std::vector<int> select_cols;
        if (offset < size) {
            uint8_t num_cols = data[offset++] % 5;
            for (uint8_t i = 0; i < num_cols && offset < size; ++i) {
                select_cols.push_back(static_cast<int>(data[offset++] % num_defaults));
            }
        }

        auto decode_csv_attrs = tensorflow::ops::DecodeCSV::Attrs()
            .FieldDelim(field_delim)
            .UseQuoteDelim(use_quote_delim)
            .NaValue(na_value)
            .SelectCols(select_cols);

        auto decode_csv_op = tensorflow::ops::DecodeCSV(root, records_input, record_defaults, decode_csv_attrs);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({decode_csv_op.output}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
