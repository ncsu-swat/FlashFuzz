#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/sparse_ops.h"
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
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10
#define MAX_NUM_SPARSE_TENSORS 3
#define MIN_NUM_SPARSE_TENSORS 2

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 11) {
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
            dtype = tensorflow::DT_BOOL;
            break;
        case 8:
            dtype = tensorflow::DT_UINT16;
            break;
        case 9:
            dtype = tensorflow::DT_UINT32;
            break;
        case 10:
            dtype = tensorflow::DT_UINT64;
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
        case tensorflow::DT_BOOL:
            fillTensorWithData<bool>(tensor, data, offset, total_size);
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
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_sparse_tensors = (data[offset++] % (MAX_NUM_SPARSE_TENSORS - MIN_NUM_SPARSE_TENSORS + 1)) + MIN_NUM_SPARSE_TENSORS;
        
        tensorflow::DataType values_dtype = parseDataType(data[offset++]);
        uint8_t rank = parseRank(data[offset++]);
        int concat_dim = static_cast<int>(data[offset++] % rank);
        
        std::vector<int64_t> base_shape = parseShape(data, offset, size, rank);
        
        std::vector<tensorflow::ops::Placeholder> indices_placeholders;
        std::vector<tensorflow::ops::Placeholder> values_placeholders;
        std::vector<tensorflow::ops::Placeholder> shapes_placeholders;
        
        std::vector<tensorflow::Tensor> indices_tensors;
        std::vector<tensorflow::Tensor> values_tensors;
        std::vector<tensorflow::Tensor> shapes_tensors;
        
        for (int i = 0; i < num_sparse_tensors; ++i) {
            if (offset >= size) break;
            
            std::vector<int64_t> current_shape = base_shape;
            if (offset < size) {
                int64_t concat_dim_size;
                if (offset + sizeof(int64_t) <= size) {
                    std::memcpy(&concat_dim_size, data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    concat_dim_size = 1 + (std::abs(concat_dim_size) % 5);
                } else {
                    concat_dim_size = 1;
                }
                current_shape[concat_dim] = concat_dim_size;
            }
            
            uint8_t num_values = (offset < size) ? (data[offset++] % 5 + 1) : 1;
            
            tensorflow::TensorShape indices_shape({num_values, rank});
            tensorflow::Tensor indices_tensor(tensorflow::DT_INT64, indices_shape);
            fillTensorWithData<int64_t>(indices_tensor, data, offset, size);
            
            auto indices_flat = indices_tensor.flat<int64_t>();
            for (int j = 0; j < indices_flat.size(); ++j) {
                int dim_idx = j % rank;
                int64_t max_val = current_shape[dim_idx] - 1;
                if (max_val >= 0) {
                    indices_flat(j) = std::abs(indices_flat(j)) % (max_val + 1);
                } else {
                    indices_flat(j) = 0;
                }
            }
            
            tensorflow::TensorShape values_shape({num_values});
            tensorflow::Tensor values_tensor(values_dtype, values_shape);
            fillTensorWithDataByType(values_tensor, values_dtype, data, offset, size);
            
            tensorflow::TensorShape shapes_shape({rank});
            tensorflow::Tensor shapes_tensor(tensorflow::DT_INT64, shapes_shape);
            auto shapes_flat = shapes_tensor.flat<int64_t>();
            for (int j = 0; j < rank; ++j) {
                shapes_flat(j) = current_shape[j];
            }
            
            indices_placeholders.push_back(tensorflow::ops::Placeholder(root, tensorflow::DT_INT64));
            values_placeholders.push_back(tensorflow::ops::Placeholder(root, values_dtype));
            shapes_placeholders.push_back(tensorflow::ops::Placeholder(root, tensorflow::DT_INT64));
            
            indices_tensors.push_back(indices_tensor);
            values_tensors.push_back(values_tensor);
            shapes_tensors.push_back(shapes_tensor);
        }
        
        if (indices_placeholders.empty()) return 0;
        
        std::vector<tensorflow::Output> indices_outputs;
        std::vector<tensorflow::Output> values_outputs;
        std::vector<tensorflow::Output> shapes_outputs;
        
        for (size_t i = 0; i < indices_placeholders.size(); ++i) {
            indices_outputs.push_back(indices_placeholders[i]);
            values_outputs.push_back(values_placeholders[i]);
            shapes_outputs.push_back(shapes_placeholders[i]);
        }
        
        auto sparse_concat = tensorflow::ops::SparseConcat(root, indices_outputs, values_outputs, shapes_outputs, concat_dim);
        
        tensorflow::ClientSession session(root);
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict;
        for (size_t i = 0; i < indices_placeholders.size(); ++i) {
            feed_dict.push_back({indices_placeholders[i].node()->name(), indices_tensors[i]});
            feed_dict.push_back({values_placeholders[i].node()->name(), values_tensors[i]});
            feed_dict.push_back({shapes_placeholders[i].node()->name(), shapes_tensors[i]});
        }
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(feed_dict, 
                                               {sparse_concat.output_indices, sparse_concat.output_values, sparse_concat.output_shape}, 
                                               {}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}