#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/dataset_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>

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
            uint8_t str_len = data[offset] % 10 + 1;
            offset++;
            
            std::string str;
            for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                str += static_cast<char>(data[offset] % 128);
                offset++;
            }
            flat(i) = tensorflow::tstring(str);
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
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::Tensor filenames_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({1}));
        fillStringTensor(filenames_tensor, data, offset, size);

        tensorflow::Tensor compression_type_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        fillStringTensor(compression_type_tensor, data, offset, size);

        tensorflow::Tensor buffer_size_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        fillTensorWithData<int64_t>(buffer_size_tensor, data, offset, size);

        tensorflow::Tensor header_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape({}));
        fillTensorWithData<bool>(header_tensor, data, offset, size);

        tensorflow::Tensor field_delim_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        fillStringTensor(field_delim_tensor, data, offset, size);

        tensorflow::Tensor use_quote_delim_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape({}));
        fillTensorWithData<bool>(use_quote_delim_tensor, data, offset, size);

        tensorflow::Tensor na_value_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        fillStringTensor(na_value_tensor, data, offset, size);

        tensorflow::Tensor select_cols_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({2}));
        fillTensorWithData<int64_t>(select_cols_tensor, data, offset, size);

        uint8_t num_defaults = (offset < size) ? (data[offset] % 3 + 1) : 1;
        offset++;

        std::vector<tensorflow::Input> record_defaults;
        std::vector<tensorflow::DataType> output_types;

        for (uint8_t i = 0; i < num_defaults; ++i) {
            tensorflow::DataType dtype = parseDataType((offset < size) ? data[offset] : 0);
            offset++;

            tensorflow::Tensor default_tensor(dtype, tensorflow::TensorShape({}));
            fillTensorWithDataByType(default_tensor, dtype, data, offset, size);
            record_defaults.push_back(tensorflow::Input(default_tensor));
            output_types.push_back(dtype);
        }

        auto filenames_input = tensorflow::Input(filenames_tensor);
        auto compression_type_input = tensorflow::Input(compression_type_tensor);
        auto buffer_size_input = tensorflow::Input(buffer_size_tensor);
        auto header_input = tensorflow::Input(header_tensor);
        auto field_delim_input = tensorflow::Input(field_delim_tensor);
        auto use_quote_delim_input = tensorflow::Input(use_quote_delim_tensor);
        auto na_value_input = tensorflow::Input(na_value_tensor);
        auto select_cols_input = tensorflow::Input(select_cols_tensor);

        // Use raw_ops approach instead of ops::ExperimentalCSVDataset
        auto experimental_csv_dataset = tensorflow::ops::experimental::CSVDatasetV2(
            root,
            filenames_input,
            compression_type_input,
            buffer_size_input,
            header_input,
            field_delim_input,
            use_quote_delim_input,
            na_value_input,
            select_cols_input,
            record_defaults,
            output_types
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({experimental_csv_dataset.handle}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
