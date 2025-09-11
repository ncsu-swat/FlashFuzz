#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
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

tensorflow::DataType parseIndexDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_INT16;
            break;
        case 1:
            dtype = tensorflow::DT_INT32;
            break;
        case 2:
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        uint8_t input_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        tensorflow::TensorShape input_tensor_shape(input_shape);
        tensorflow::Tensor input_tensor(input_dtype, input_tensor_shape);
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        
        tensorflow::DataType index_dtype = parseIndexDataType(data[offset++]);
        
        uint8_t slice_dims = (offset < size) ? (data[offset++] % 4 + 1) : 1;
        
        std::vector<int32_t> begin_values, end_values, strides_values;
        for (uint8_t i = 0; i < slice_dims; ++i) {
            int32_t begin_val = (offset + 4 <= size) ? 
                *reinterpret_cast<const int32_t*>(data + offset) : 0;
            offset += 4;
            begin_values.push_back(begin_val % 10);
            
            int32_t end_val = (offset + 4 <= size) ? 
                *reinterpret_cast<const int32_t*>(data + offset) : 1;
            offset += 4;
            end_values.push_back(end_val % 10);
            
            int32_t stride_val = (offset + 4 <= size) ? 
                *reinterpret_cast<const int32_t*>(data + offset) : 1;
            offset += 4;
            if (stride_val == 0) stride_val = 1;
            strides_values.push_back(stride_val);
        }
        
        tensorflow::TensorShape slice_shape({static_cast<int64_t>(slice_dims)});
        tensorflow::Tensor begin_tensor(index_dtype, slice_shape);
        tensorflow::Tensor end_tensor(index_dtype, slice_shape);
        tensorflow::Tensor strides_tensor(index_dtype, slice_shape);
        
        if (index_dtype == tensorflow::DT_INT16) {
            auto begin_flat = begin_tensor.flat<int16_t>();
            auto end_flat = end_tensor.flat<int16_t>();
            auto strides_flat = strides_tensor.flat<int16_t>();
            for (int i = 0; i < slice_dims; ++i) {
                begin_flat(i) = static_cast<int16_t>(begin_values[i]);
                end_flat(i) = static_cast<int16_t>(end_values[i]);
                strides_flat(i) = static_cast<int16_t>(strides_values[i]);
            }
        } else if (index_dtype == tensorflow::DT_INT32) {
            auto begin_flat = begin_tensor.flat<int32_t>();
            auto end_flat = end_tensor.flat<int32_t>();
            auto strides_flat = strides_tensor.flat<int32_t>();
            for (int i = 0; i < slice_dims; ++i) {
                begin_flat(i) = begin_values[i];
                end_flat(i) = end_values[i];
                strides_flat(i) = strides_values[i];
            }
        } else {
            auto begin_flat = begin_tensor.flat<int64_t>();
            auto end_flat = end_tensor.flat<int64_t>();
            auto strides_flat = strides_tensor.flat<int64_t>();
            for (int i = 0; i < slice_dims; ++i) {
                begin_flat(i) = static_cast<int64_t>(begin_values[i]);
                end_flat(i) = static_cast<int64_t>(end_values[i]);
                strides_flat(i) = static_cast<int64_t>(strides_values[i]);
            }
        }
        
        int32_t begin_mask = (offset < size) ? data[offset++] : 0;
        int32_t end_mask = (offset < size) ? data[offset++] : 0;
        int32_t ellipsis_mask = (offset < size) ? (data[offset++] & 1) : 0;
        int32_t new_axis_mask = (offset < size) ? data[offset++] : 0;
        int32_t shrink_axis_mask = (offset < size) ? data[offset++] : 0;
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, input_dtype);
        auto begin_placeholder = tensorflow::ops::Placeholder(root, index_dtype);
        auto end_placeholder = tensorflow::ops::Placeholder(root, index_dtype);
        auto strides_placeholder = tensorflow::ops::Placeholder(root, index_dtype);
        
        auto strided_slice_op = tensorflow::ops::StridedSlice(
            root, input_placeholder, begin_placeholder, end_placeholder, strides_placeholder,
            tensorflow::ops::StridedSlice::Attrs()
                .BeginMask(begin_mask)
                .EndMask(end_mask)
                .EllipsisMask(ellipsis_mask)
                .NewAxisMask(new_axis_mask)
                .ShrinkAxisMask(shrink_axis_mask)
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({
            {input_placeholder, input_tensor},
            {begin_placeholder, begin_tensor},
            {end_placeholder, end_tensor},
            {strides_placeholder, strides_tensor}
        }, {strided_slice_op}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
