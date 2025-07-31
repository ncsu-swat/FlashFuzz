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
            dtype = tensorflow::DT_STRING;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
            break;
        case 2:
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

void fillTensorWithDataByType(tensorflow::Tensor& tensor,
                              tensorflow::DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
    switch (dtype) {
        case tensorflow::DT_STRING:
            fillStringTensor(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
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
        uint8_t filenames_rank = parseRank(data[offset++]);
        std::vector<int64_t> filenames_shape = parseShape(data, offset, size, filenames_rank);
        tensorflow::Tensor filenames_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(filenames_shape));
        fillTensorWithDataByType(filenames_tensor, tensorflow::DT_STRING, data, offset, size);

        uint8_t header_bytes_rank = parseRank(data[offset++]);
        std::vector<int64_t> header_bytes_shape = parseShape(data, offset, size, header_bytes_rank);
        tensorflow::Tensor header_bytes_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(header_bytes_shape));
        fillTensorWithDataByType(header_bytes_tensor, tensorflow::DT_INT64, data, offset, size);

        uint8_t record_bytes_rank = parseRank(data[offset++]);
        std::vector<int64_t> record_bytes_shape = parseShape(data, offset, size, record_bytes_rank);
        tensorflow::Tensor record_bytes_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(record_bytes_shape));
        fillTensorWithDataByType(record_bytes_tensor, tensorflow::DT_INT64, data, offset, size);

        uint8_t footer_bytes_rank = parseRank(data[offset++]);
        std::vector<int64_t> footer_bytes_shape = parseShape(data, offset, size, footer_bytes_rank);
        tensorflow::Tensor footer_bytes_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(footer_bytes_shape));
        fillTensorWithDataByType(footer_bytes_tensor, tensorflow::DT_INT64, data, offset, size);

        uint8_t buffer_size_rank = parseRank(data[offset++]);
        std::vector<int64_t> buffer_size_shape = parseShape(data, offset, size, buffer_size_rank);
        tensorflow::Tensor buffer_size_tensor(tensorflow::DT_INT64, tensorflow::TensorShape(buffer_size_shape));
        fillTensorWithDataByType(buffer_size_tensor, tensorflow::DT_INT64, data, offset, size);

        uint8_t compression_type_rank = parseRank(data[offset++]);
        std::vector<int64_t> compression_type_shape = parseShape(data, offset, size, compression_type_rank);
        tensorflow::Tensor compression_type_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(compression_type_shape));
        fillTensorWithDataByType(compression_type_tensor, tensorflow::DT_STRING, data, offset, size);

        auto filenames_input = tensorflow::ops::Const(root, filenames_tensor);
        auto header_bytes_input = tensorflow::ops::Const(root, header_bytes_tensor);
        auto record_bytes_input = tensorflow::ops::Const(root, record_bytes_tensor);
        auto footer_bytes_input = tensorflow::ops::Const(root, footer_bytes_tensor);
        auto buffer_size_input = tensorflow::ops::Const(root, buffer_size_tensor);
        auto compression_type_input = tensorflow::ops::Const(root, compression_type_tensor);

        // Use the raw_ops namespace to access FixedLengthRecordDatasetV2
        auto dataset_op = tensorflow::ops::FixedLengthRecordDataset(
            root,
            filenames_input,
            record_bytes_input,
            header_bytes_input,
            footer_bytes_input,
            buffer_size_input,
            compression_type_input
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({dataset_op}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}