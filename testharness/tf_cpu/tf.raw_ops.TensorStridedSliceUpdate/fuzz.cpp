#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
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
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 15) {
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
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 9:
            dtype = tensorflow::DT_UINT16;
            break;
        case 10:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 11:
            dtype = tensorflow::DT_HALF;
            break;
        case 12:
            dtype = tensorflow::DT_UINT32;
            break;
        case 13:
            dtype = tensorflow::DT_UINT64;
            break;
        case 14:
            dtype = tensorflow::DT_COMPLEX128;
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
        case tensorflow::DT_BFLOAT16:
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_COMPLEX64:
            fillTensorWithData<tensorflow::complex64>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_COMPLEX128:
            fillTensorWithData<tensorflow::complex128>(tensor, data, offset, total_size);
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
        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        uint8_t input_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        tensorflow::TensorShape input_tensor_shape(input_shape);
        tensorflow::Tensor input_tensor(input_dtype, input_tensor_shape);
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        
        uint8_t slice_rank = parseRank(data[offset++]);
        std::vector<int64_t> slice_shape = {slice_rank};
        
        tensorflow::TensorShape slice_tensor_shape(slice_shape);
        tensorflow::Tensor begin_tensor(tensorflow::DT_INT32, slice_tensor_shape);
        tensorflow::Tensor end_tensor(tensorflow::DT_INT32, slice_tensor_shape);
        tensorflow::Tensor strides_tensor(tensorflow::DT_INT32, slice_tensor_shape);
        
        auto begin_flat = begin_tensor.flat<int32_t>();
        auto end_flat = end_tensor.flat<int32_t>();
        auto strides_flat = strides_tensor.flat<int32_t>();
        
        for (int i = 0; i < slice_rank; ++i) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t begin_val, end_val, stride_val;
                std::memcpy(&begin_val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                std::memcpy(&end_val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                std::memcpy(&stride_val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                
                if (i < input_shape.size()) {
                    begin_val = std::abs(begin_val) % static_cast<int32_t>(input_shape[i]);
                    end_val = begin_val + 1 + (std::abs(end_val) % (static_cast<int32_t>(input_shape[i]) - begin_val));
                    stride_val = (stride_val == 0) ? 1 : std::abs(stride_val);
                } else {
                    begin_val = 0;
                    end_val = 1;
                    stride_val = 1;
                }
                
                begin_flat(i) = begin_val;
                end_flat(i) = end_val;
                strides_flat(i) = stride_val;
            } else {
                begin_flat(i) = 0;
                end_flat(i) = 1;
                strides_flat(i) = 1;
            }
        }
        
        std::vector<int64_t> value_shape;
        for (int i = 0; i < slice_rank && i < input_shape.size(); ++i) {
            int64_t slice_size = (end_flat(i) - begin_flat(i) + strides_flat(i) - 1) / strides_flat(i);
            value_shape.push_back(slice_size);
        }
        
        if (value_shape.empty()) {
            value_shape.push_back(1);
        }
        
        tensorflow::TensorShape value_tensor_shape(value_shape);
        tensorflow::Tensor value_tensor(input_dtype, value_tensor_shape);
        fillTensorWithDataByType(value_tensor, input_dtype, data, offset, size);
        
        int begin_mask = 0;
        int end_mask = 0;
        int ellipsis_mask = 0;
        int new_axis_mask = 0;
        int shrink_axis_mask = 0;
        
        if (offset < size) {
            begin_mask = data[offset++] % 256;
        }
        if (offset < size) {
            end_mask = data[offset++] % 256;
        }
        if (offset < size) {
            ellipsis_mask = data[offset++] % 256;
        }
        if (offset < size) {
            new_axis_mask = data[offset++] % 256;
        }
        if (offset < size) {
            shrink_axis_mask = data[offset++] % 256;
        }
        
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto begin_op = tensorflow::ops::Const(root, begin_tensor);
        auto end_op = tensorflow::ops::Const(root, end_tensor);
        auto strides_op = tensorflow::ops::Const(root, strides_tensor);
        auto value_op = tensorflow::ops::Const(root, value_tensor);
        
        auto tensor_strided_slice_update = tensorflow::ops::TensorStridedSliceUpdate(
            root, input_op, begin_op, end_op, strides_op, value_op,
            tensorflow::ops::TensorStridedSliceUpdate::BeginMask(begin_mask)
                .EndMask(end_mask)
                .EllipsisMask(ellipsis_mask)
                .NewAxisMask(new_axis_mask)
                .ShrinkAxisMask(shrink_axis_mask)
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({tensor_strided_slice_update}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}