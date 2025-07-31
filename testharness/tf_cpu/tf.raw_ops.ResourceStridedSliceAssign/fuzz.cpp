#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType value_dtype = parseDataType(data[offset++]);
        uint8_t value_rank = parseRank(data[offset++]);
        std::vector<int64_t> value_shape = parseShape(data, offset, size, value_rank);
        
        tensorflow::DataType index_dtype = (data[offset++] % 2 == 0) ? tensorflow::DT_INT32 : tensorflow::DT_INT64;
        uint8_t index_rank = 1;
        std::vector<int64_t> index_shape = {static_cast<int64_t>(value_rank)};
        
        if (value_rank == 0) {
            index_shape = {1};
        }

        tensorflow::Tensor value_tensor(value_dtype, tensorflow::TensorShape(value_shape));
        fillTensorWithDataByType(value_tensor, value_dtype, data, offset, size);

        tensorflow::Tensor begin_tensor(index_dtype, tensorflow::TensorShape(index_shape));
        fillTensorWithDataByType(begin_tensor, index_dtype, data, offset, size);

        tensorflow::Tensor end_tensor(index_dtype, tensorflow::TensorShape(index_shape));
        fillTensorWithDataByType(end_tensor, index_dtype, data, offset, size);

        tensorflow::Tensor strides_tensor(index_dtype, tensorflow::TensorShape(index_shape));
        fillTensorWithDataByType(strides_tensor, index_dtype, data, offset, size);

        if (index_dtype == tensorflow::DT_INT32) {
            auto strides_flat = strides_tensor.flat<int32_t>();
            for (int i = 0; i < strides_flat.size(); ++i) {
                if (strides_flat(i) == 0) strides_flat(i) = 1;
            }
        } else {
            auto strides_flat = strides_tensor.flat<int64_t>();
            for (int i = 0; i < strides_flat.size(); ++i) {
                if (strides_flat(i) == 0) strides_flat(i) = 1;
            }
        }

        auto var = tensorflow::ops::VarHandleOp(root, value_dtype, tensorflow::TensorShape(value_shape));
        auto init_var = tensorflow::ops::AssignVariableOp(root, var, tensorflow::ops::Const(root, value_tensor));

        int32_t begin_mask = 0;
        int32_t end_mask = 0;
        int32_t ellipsis_mask = 0;
        int32_t new_axis_mask = 0;
        int32_t shrink_axis_mask = 0;

        if (offset < size) begin_mask = data[offset++] % 256;
        if (offset < size) end_mask = data[offset++] % 256;
        if (offset < size) ellipsis_mask = data[offset++] % 256;
        if (offset < size) new_axis_mask = data[offset++] % 256;
        if (offset < size) shrink_axis_mask = data[offset++] % 256;

        auto strided_slice_assign = tensorflow::ops::ResourceStridedSliceAssign(
            root, 
            var, 
            tensorflow::ops::Const(root, begin_tensor),
            tensorflow::ops::Const(root, end_tensor),
            tensorflow::ops::Const(root, strides_tensor),
            tensorflow::ops::Const(root, value_tensor),
            tensorflow::ops::ResourceStridedSliceAssign::BeginMask(begin_mask)
                .EndMask(end_mask)
                .EllipsisMask(ellipsis_mask)
                .NewAxisMask(new_axis_mask)
                .ShrinkAxisMask(shrink_axis_mask)
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Operation> ops_to_run = {init_var, strided_slice_assign.operation};
        tensorflow::Status status = session.Run(ops_to_run, nullptr);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}