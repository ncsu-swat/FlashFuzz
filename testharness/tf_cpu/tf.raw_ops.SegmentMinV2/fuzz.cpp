#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>

#define MAX_RANK 4
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 12) {
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
            dtype = tensorflow::DT_UINT8;
            break;
        case 4:
            dtype = tensorflow::DT_INT16;
            break;
        case 5:
            dtype = tensorflow::DT_INT8;
            break;
        case 6:
            dtype = tensorflow::DT_INT64;
            break;
        case 7:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 8:
            dtype = tensorflow::DT_UINT16;
            break;
        case 9:
            dtype = tensorflow::DT_HALF;
            break;
        case 10:
            dtype = tensorflow::DT_UINT32;
            break;
        case 11:
            dtype = tensorflow::DT_UINT64;
            break;
    }
    return dtype;
}

tensorflow::DataType parseSegmentIdsDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
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
        case tensorflow::DT_UINT8:
            fillTensorWithData<uint8_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT16:
            fillTensorWithData<int16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT8:
            fillTensorWithData<int8_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT16:
            fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT32:
            fillTensorWithData<uint32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT64:
            fillTensorWithData<uint64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BFLOAT16:
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType data_dtype = parseDataType(data[offset++]);
        tensorflow::DataType segment_ids_dtype = parseSegmentIdsDataType(data[offset++]);
        tensorflow::DataType num_segments_dtype = parseSegmentIdsDataType(data[offset++]);
        
        uint8_t data_rank = parseRank(data[offset++]);
        std::vector<int64_t> data_shape = parseShape(data, offset, size, data_rank);
        
        if (data_shape.empty() || data_shape[0] <= 0) {
            return 0;
        }
        
        int64_t first_dim = data_shape[0];
        
        tensorflow::TensorShape data_tensor_shape;
        for (int64_t dim : data_shape) {
            data_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor data_tensor(data_dtype, data_tensor_shape);
        fillTensorWithDataByType(data_tensor, data_dtype, data, offset, size);
        
        tensorflow::TensorShape segment_ids_shape;
        segment_ids_shape.AddDim(first_dim);
        tensorflow::Tensor segment_ids_tensor(segment_ids_dtype, segment_ids_shape);
        
        if (segment_ids_dtype == tensorflow::DT_INT32) {
            auto flat = segment_ids_tensor.flat<int32_t>();
            for (int64_t i = 0; i < first_dim; ++i) {
                if (offset < size) {
                    int32_t val;
                    std::memcpy(&val, data + offset, sizeof(int32_t));
                    offset += sizeof(int32_t);
                    flat(i) = std::abs(val) % std::max(1, static_cast<int>(first_dim));
                } else {
                    flat(i) = static_cast<int32_t>(i % std::max(1, static_cast<int>(first_dim)));
                }
            }
        } else {
            auto flat = segment_ids_tensor.flat<int64_t>();
            for (int64_t i = 0; i < first_dim; ++i) {
                if (offset < size) {
                    int64_t val;
                    std::memcpy(&val, data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    flat(i) = std::abs(val) % std::max(1L, first_dim);
                } else {
                    flat(i) = i % std::max(1L, first_dim);
                }
            }
        }
        
        int64_t num_segments_val = first_dim;
        if (offset < size) {
            int64_t val;
            std::memcpy(&val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_segments_val = std::abs(val) % (first_dim + 5) + 1;
        }
        
        tensorflow::TensorShape num_segments_shape;
        tensorflow::Tensor num_segments_tensor(num_segments_dtype, num_segments_shape);
        
        if (num_segments_dtype == tensorflow::DT_INT32) {
            num_segments_tensor.scalar<int32_t>()() = static_cast<int32_t>(num_segments_val);
        } else {
            num_segments_tensor.scalar<int64_t>()() = num_segments_val;
        }
        
        auto data_input = tensorflow::ops::Placeholder(root, data_dtype);
        auto segment_ids_input = tensorflow::ops::Placeholder(root, segment_ids_dtype);
        auto num_segments_input = tensorflow::ops::Placeholder(root, num_segments_dtype);
        
        auto segment_min_v2 = tensorflow::ops::SegmentMinV2(root, data_input, segment_ids_input, num_segments_input);
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({{data_input, data_tensor}, 
                                                 {segment_ids_input, segment_ids_tensor},
                                                 {num_segments_input, num_segments_tensor}}, 
                                                {segment_min_v2}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}