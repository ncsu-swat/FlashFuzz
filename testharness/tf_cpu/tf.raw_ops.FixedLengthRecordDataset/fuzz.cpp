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
            size_t str_len = std::min(static_cast<size_t>(10), total_size - offset);
            std::string str(reinterpret_cast<const char*>(data + offset), str_len);
            offset += str_len;
            flat(i) = str;
        } else {
            flat(i) = "";
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t filenames_rank = parseRank(data[offset++]);
        std::vector<int64_t> filenames_shape = parseShape(data, offset, size, filenames_rank);
        tensorflow::Tensor filenames_tensor(tensorflow::DT_STRING, tensorflow::TensorShape(filenames_shape));
        fillStringTensor(filenames_tensor, data, offset, size);
        
        tensorflow::Tensor header_bytes_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        if (offset + sizeof(int64_t) <= size) {
            int64_t header_val;
            std::memcpy(&header_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            header_val = std::abs(header_val) % 1000;
            header_bytes_tensor.scalar<int64_t>()() = header_val;
        } else {
            header_bytes_tensor.scalar<int64_t>()() = 0;
        }

        tensorflow::Tensor record_bytes_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        if (offset + sizeof(int64_t) <= size) {
            int64_t record_val;
            std::memcpy(&record_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            record_val = (std::abs(record_val) % 1000 + 1);
            if (record_val < 1LL) record_val = 1LL;
            record_bytes_tensor.scalar<int64_t>()() = record_val;
        } else {
            record_bytes_tensor.scalar<int64_t>()() = 1;
        }

        tensorflow::Tensor footer_bytes_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        if (offset + sizeof(int64_t) <= size) {
            int64_t footer_val;
            std::memcpy(&footer_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            footer_val = std::abs(footer_val) % 1000;
            footer_bytes_tensor.scalar<int64_t>()() = footer_val;
        } else {
            footer_bytes_tensor.scalar<int64_t>()() = 0;
        }

        tensorflow::Tensor buffer_size_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        if (offset + sizeof(int64_t) <= size) {
            int64_t buffer_val;
            std::memcpy(&buffer_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            buffer_val = (std::abs(buffer_val) % 10000 + 1);
            if (buffer_val < 1LL) buffer_val = 1LL;
            buffer_size_tensor.scalar<int64_t>()() = buffer_val;
        } else {
            buffer_size_tensor.scalar<int64_t>()() = 1;
        }

        auto filenames_op = tensorflow::ops::Const(root, filenames_tensor);
        auto header_bytes_op = tensorflow::ops::Const(root, header_bytes_tensor);
        auto record_bytes_op = tensorflow::ops::Const(root, record_bytes_tensor);
        auto footer_bytes_op = tensorflow::ops::Const(root, footer_bytes_tensor);
        auto buffer_size_op = tensorflow::ops::Const(root, buffer_size_tensor);

        // Use raw_ops namespace for FixedLengthRecordDataset
        auto dataset = tensorflow::ops::FixedLengthRecordDatasetV2(
            root, 
            filenames_op, 
            header_bytes_op, 
            record_bytes_op, 
            footer_bytes_op, 
            buffer_size_op
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({dataset.handle}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
