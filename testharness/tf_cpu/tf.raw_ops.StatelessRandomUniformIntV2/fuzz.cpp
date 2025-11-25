#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/random_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/graph/node_builder.h"
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
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseShapeDataType(uint8_t selector) {
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

tensorflow::DataType parseMinMaxDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
            break;
        case 2:
            dtype = tensorflow::DT_UINT32;
            break;
        case 3:
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
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
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
        uint8_t shape_dtype_selector = data[offset++];
        tensorflow::DataType shape_dtype = parseShapeDataType(shape_dtype_selector);
        
        uint8_t minmax_dtype_selector = data[offset++];
        tensorflow::DataType minmax_dtype = parseMinMaxDataType(minmax_dtype_selector);
        
        uint8_t shape_rank_byte = data[offset++];
        uint8_t shape_rank = parseRank(shape_rank_byte);
        
        std::vector<int64_t> output_shape = parseShape(data, offset, size, shape_rank);
        
        tensorflow::TensorShape shape_tensor_shape({static_cast<int64_t>(output_shape.size())});
        tensorflow::Tensor shape_tensor(shape_dtype, shape_tensor_shape);
        
        if (shape_dtype == tensorflow::DT_INT32) {
            auto flat = shape_tensor.flat<int32_t>();
            for (size_t i = 0; i < output_shape.size(); ++i) {
                flat(i) = static_cast<int32_t>(output_shape[i]);
            }
        } else {
            auto flat = shape_tensor.flat<int64_t>();
            for (size_t i = 0; i < output_shape.size(); ++i) {
                flat(i) = output_shape[i];
            }
        }
        
        tensorflow::TensorShape key_shape({1});
        tensorflow::Tensor key_tensor(tensorflow::DT_UINT64, key_shape);
        fillTensorWithData<uint64_t>(key_tensor, data, offset, size);
        
        tensorflow::TensorShape counter_shape({2});
        tensorflow::Tensor counter_tensor(tensorflow::DT_UINT64, counter_shape);
        fillTensorWithData<uint64_t>(counter_tensor, data, offset, size);
        
        tensorflow::TensorShape alg_shape({});
        tensorflow::Tensor alg_tensor(tensorflow::DT_INT32, alg_shape);
        fillTensorWithData<int32_t>(alg_tensor, data, offset, size);
        int32_t alg_val = alg_tensor.scalar<int32_t>()();
        alg_tensor.scalar<int32_t>()() = std::abs(alg_val % 3) + 1;
        
        tensorflow::TensorShape minval_shape({});
        tensorflow::Tensor minval_tensor(minmax_dtype, minval_shape);
        fillTensorWithDataByType(minval_tensor, minmax_dtype, data, offset, size);
        
        tensorflow::TensorShape maxval_shape({});
        tensorflow::Tensor maxval_tensor(minmax_dtype, maxval_shape);
        fillTensorWithDataByType(maxval_tensor, minmax_dtype, data, offset, size);
        
        // Normalize min/max to a valid range for integer generation.
        switch (minmax_dtype) {
            case tensorflow::DT_INT32: {
                int32_t min_v = minval_tensor.scalar<int32_t>()();
                int32_t max_v = maxval_tensor.scalar<int32_t>()();
                if (max_v <= min_v) {
                    min_v = 0;
                    max_v = 1;
                }
                minval_tensor.scalar<int32_t>()() = min_v;
                maxval_tensor.scalar<int32_t>()() = max_v;
                break;
            }
            case tensorflow::DT_INT64: {
                int64_t min_v = minval_tensor.scalar<int64_t>()();
                int64_t max_v = maxval_tensor.scalar<int64_t>()();
                if (max_v <= min_v) {
                    min_v = 0;
                    max_v = 1;
                }
                minval_tensor.scalar<int64_t>()() = min_v;
                maxval_tensor.scalar<int64_t>()() = max_v;
                break;
            }
            case tensorflow::DT_UINT32: {
                uint32_t min_v = minval_tensor.scalar<uint32_t>()();
                uint32_t max_v = maxval_tensor.scalar<uint32_t>()();
                if (max_v <= min_v) {
                    min_v = 0;
                    max_v = 1;
                }
                minval_tensor.scalar<uint32_t>()() = min_v;
                maxval_tensor.scalar<uint32_t>()() = max_v;
                break;
            }
            case tensorflow::DT_UINT64: {
                uint64_t min_v = minval_tensor.scalar<uint64_t>()();
                uint64_t max_v = maxval_tensor.scalar<uint64_t>()();
                if (max_v <= min_v) {
                    min_v = 0;
                    max_v = 1;
                }
                minval_tensor.scalar<uint64_t>()() = min_v;
                maxval_tensor.scalar<uint64_t>()() = max_v;
                break;
            }
            default:
                break;
        }

        auto shape_input = tensorflow::ops::Const(root, shape_tensor);
        auto key_input = tensorflow::ops::Const(root, key_tensor);
        auto counter_input = tensorflow::ops::Const(root, counter_tensor);
        auto alg_input = tensorflow::ops::Const(root, alg_tensor);
        auto minval_input = tensorflow::ops::Const(root, minval_tensor);
        auto maxval_input = tensorflow::ops::Const(root, maxval_tensor);

        tensorflow::Node* node = nullptr;
        tensorflow::Status status = tensorflow::NodeBuilder(
                                        root.GetUniqueNameForOp("stateless_random_uniform_int_v2"),
                                        "StatelessRandomUniformIntV2")
                                        .Input(tensorflow::NodeBuilder::NodeOut(shape_input.node()))
                                        .Input(tensorflow::NodeBuilder::NodeOut(key_input.node()))
                                        .Input(tensorflow::NodeBuilder::NodeOut(counter_input.node()))
                                        .Input(tensorflow::NodeBuilder::NodeOut(alg_input.node()))
                                        .Input(tensorflow::NodeBuilder::NodeOut(minval_input.node()))
                                        .Input(tensorflow::NodeBuilder::NodeOut(maxval_input.node()))
                                        .Attr("dtype", minmax_dtype)
                                        .Attr("Tshape", shape_tensor.dtype())
                                        .Finalize(root.graph(), &node);
        if (!status.ok()) {
            return -1;
        }

        tensorflow::Output result(node, 0);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({result}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
