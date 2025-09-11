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

tensorflow::DataType parsePartitionDataType(uint8_t selector) {
    return (selector % 2 == 0) ? tensorflow::DT_INT32 : tensorflow::DT_INT64;
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
        tensorflow::DataType values_dtype = parseDataType(data[offset++]);
        tensorflow::DataType partition_dtype = parsePartitionDataType(data[offset++]);
        
        uint8_t num_partitions = (data[offset++] % 3) + 1;
        
        uint8_t values_size = (data[offset++] % 10) + 1;
        tensorflow::Tensor values_tensor(values_dtype, tensorflow::TensorShape({values_size}));
        fillTensorWithDataByType(values_tensor, values_dtype, data, offset, size);
        
        tensorflow::Tensor default_value_tensor(values_dtype, tensorflow::TensorShape({}));
        fillTensorWithDataByType(default_value_tensor, values_dtype, data, offset, size);
        
        std::vector<tensorflow::Tensor> partition_tensors;
        std::vector<std::string> partition_types;
        
        for (uint8_t i = 0; i < num_partitions; ++i) {
            uint8_t partition_size = (data[offset % size] % 10) + 2;
            offset++;
            
            tensorflow::Tensor partition_tensor(partition_dtype, tensorflow::TensorShape({partition_size}));
            
            if (partition_dtype == tensorflow::DT_INT32) {
                auto flat = partition_tensor.flat<int32_t>();
                for (int j = 0; j < partition_size; ++j) {
                    if (offset < size) {
                        int32_t val = static_cast<int32_t>(data[offset++] % (values_size + 1));
                        flat(j) = (j == 0) ? 0 : std::max(flat(j-1), val);
                    } else {
                        flat(j) = (j == 0) ? 0 : flat(j-1);
                    }
                }
            } else {
                auto flat = partition_tensor.flat<int64_t>();
                for (int j = 0; j < partition_size; ++j) {
                    if (offset < size) {
                        int64_t val = static_cast<int64_t>(data[offset++] % (values_size + 1));
                        flat(j) = (j == 0) ? 0 : std::max(flat(j-1), val);
                    } else {
                        flat(j) = (j == 0) ? 0 : flat(j-1);
                    }
                }
            }
            
            partition_tensors.push_back(partition_tensor);
            partition_types.push_back("ROW_SPLITS");
        }
        
        uint8_t shape_rank = parseRank(data[offset % size]);
        offset++;
        std::vector<int64_t> shape_dims = parseShape(data, offset, size, shape_rank);
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({static_cast<int64_t>(shape_dims.size())}));
        auto shape_flat = shape_tensor.flat<int64_t>();
        for (size_t i = 0; i < shape_dims.size(); ++i) {
            shape_flat(i) = shape_dims[i];
        }
        
        auto shape_input = tensorflow::ops::Const(root, shape_tensor);
        auto values_input = tensorflow::ops::Const(root, values_tensor);
        auto default_value_input = tensorflow::ops::Const(root, default_value_tensor);
        
        std::vector<tensorflow::Output> partition_inputs;
        for (const auto& tensor : partition_tensors) {
            partition_inputs.push_back(tensorflow::ops::Const(root, tensor));
        }
        
        // Use raw_ops namespace to access RaggedTensorToTensor
        auto ragged_to_tensor_op = tensorflow::ops::RaggedTensorToSparse(
            root,
            partition_inputs,
            values_input,
            partition_types
        );
        
        // Convert sparse back to dense as a workaround
        auto sparse_to_dense = tensorflow::ops::SparseToDense(
            root.WithOpName("SparseToDense"),
            ragged_to_tensor_op.indices,
            shape_input,
            ragged_to_tensor_op.values,
            default_value_input
        );
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({sparse_to_dense}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
