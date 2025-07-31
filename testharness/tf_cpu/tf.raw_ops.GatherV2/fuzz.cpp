#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <cstring>
#include <vector>
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

tensorflow::DataType parseIndicesDataType(uint8_t selector) {
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

tensorflow::DataType parseAxisDataType(uint8_t selector) {
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
        tensorflow::DataType params_dtype = parseDataType(data[offset++]);
        uint8_t params_rank = parseRank(data[offset++]);
        std::vector<int64_t> params_shape = parseShape(data, offset, size, params_rank);
        
        tensorflow::DataType indices_dtype = parseIndicesDataType(data[offset++]);
        uint8_t indices_rank = parseRank(data[offset++]);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        
        tensorflow::DataType axis_dtype = parseAxisDataType(data[offset++]);
        
        int32_t batch_dims = 0;
        if (offset < size) {
            batch_dims = static_cast<int32_t>(data[offset++] % 3);
        }

        tensorflow::TensorShape params_tensor_shape(params_shape);
        tensorflow::Tensor params_tensor(params_dtype, params_tensor_shape);
        fillTensorWithDataByType(params_tensor, params_dtype, data, offset, size);

        tensorflow::TensorShape indices_tensor_shape(indices_shape);
        tensorflow::Tensor indices_tensor(indices_dtype, indices_tensor_shape);
        
        if (indices_dtype == tensorflow::DT_INT16) {
            auto flat = indices_tensor.flat<int16_t>();
            for (int i = 0; i < flat.size(); ++i) {
                if (offset + sizeof(int16_t) <= size) {
                    int16_t value;
                    std::memcpy(&value, data + offset, sizeof(int16_t));
                    offset += sizeof(int16_t);
                    if (params_rank > 0) {
                        flat(i) = std::abs(value) % static_cast<int16_t>(params_shape[0]);
                    } else {
                        flat(i) = 0;
                    }
                } else {
                    flat(i) = 0;
                }
            }
        } else if (indices_dtype == tensorflow::DT_INT32) {
            auto flat = indices_tensor.flat<int32_t>();
            for (int i = 0; i < flat.size(); ++i) {
                if (offset + sizeof(int32_t) <= size) {
                    int32_t value;
                    std::memcpy(&value, data + offset, sizeof(int32_t));
                    offset += sizeof(int32_t);
                    if (params_rank > 0) {
                        flat(i) = std::abs(value) % static_cast<int32_t>(params_shape[0]);
                    } else {
                        flat(i) = 0;
                    }
                } else {
                    flat(i) = 0;
                }
            }
        } else if (indices_dtype == tensorflow::DT_INT64) {
            auto flat = indices_tensor.flat<int64_t>();
            for (int i = 0; i < flat.size(); ++i) {
                if (offset + sizeof(int64_t) <= size) {
                    int64_t value;
                    std::memcpy(&value, data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    if (params_rank > 0) {
                        flat(i) = std::abs(value) % params_shape[0];
                    } else {
                        flat(i) = 0;
                    }
                } else {
                    flat(i) = 0;
                }
            }
        }

        tensorflow::Tensor axis_tensor;
        if (axis_dtype == tensorflow::DT_INT32) {
            axis_tensor = tensorflow::Tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
            int32_t axis_value = 0;
            if (offset + sizeof(int32_t) <= size) {
                std::memcpy(&axis_value, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
            }
            if (params_rank > 0) {
                axis_value = axis_value % static_cast<int32_t>(params_rank);
                if (axis_value < 0) axis_value += static_cast<int32_t>(params_rank);
            } else {
                axis_value = 0;
            }
            axis_tensor.scalar<int32_t>()() = axis_value;
        } else {
            axis_tensor = tensorflow::Tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
            int64_t axis_value = 0;
            if (offset + sizeof(int64_t) <= size) {
                std::memcpy(&axis_value, data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            }
            if (params_rank > 0) {
                axis_value = axis_value % static_cast<int64_t>(params_rank);
                if (axis_value < 0) axis_value += static_cast<int64_t>(params_rank);
            } else {
                axis_value = 0;
            }
            axis_tensor.scalar<int64_t>()() = axis_value;
        }

        auto params_placeholder = tensorflow::ops::Placeholder(root, params_dtype);
        auto indices_placeholder = tensorflow::ops::Placeholder(root, indices_dtype);
        auto axis_placeholder = tensorflow::ops::Placeholder(root, axis_dtype);

        auto gather_op = tensorflow::ops::GatherV2(root, params_placeholder, indices_placeholder, axis_placeholder, 
                                                   tensorflow::ops::GatherV2::BatchDims(batch_dims));

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;

        tensorflow::Status status = session.Run({{params_placeholder, params_tensor}, 
                                                  {indices_placeholder, indices_tensor}, 
                                                  {axis_placeholder, axis_tensor}}, 
                                                 {gather_op}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}