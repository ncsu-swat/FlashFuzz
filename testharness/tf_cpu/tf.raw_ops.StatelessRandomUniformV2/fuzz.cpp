#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/random_ops.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/framework/types.h"
#include <cstring>
#include <vector>
#include <iostream>

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
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 2:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 3:
            dtype = tensorflow::DT_DOUBLE;
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType output_dtype = parseDataType(data[offset++]);
        
        uint8_t shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> output_shape = parseShape(data, offset, size, shape_rank);
        
        tensorflow::TensorShape shape_tensor_shape({static_cast<int64_t>(output_shape.size())});
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT64, shape_tensor_shape);
        auto shape_flat = shape_tensor.flat<int64_t>();
        for (size_t i = 0; i < output_shape.size(); ++i) {
            shape_flat(i) = output_shape[i];
        }
        
        tensorflow::TensorShape key_shape({1});
        tensorflow::Tensor key_tensor(tensorflow::DT_UINT64, key_shape);
        if (offset + sizeof(uint64_t) <= size) {
            uint64_t key_val;
            std::memcpy(&key_val, data + offset, sizeof(uint64_t));
            offset += sizeof(uint64_t);
            key_tensor.flat<uint64_t>()(0) = key_val;
        } else {
            key_tensor.flat<uint64_t>()(0) = 12345;
        }
        
        tensorflow::TensorShape counter_shape({2});
        tensorflow::Tensor counter_tensor(tensorflow::DT_UINT64, counter_shape);
        auto counter_flat = counter_tensor.flat<uint64_t>();
        for (int i = 0; i < 2; ++i) {
            if (offset + sizeof(uint64_t) <= size) {
                uint64_t counter_val;
                std::memcpy(&counter_val, data + offset, sizeof(uint64_t));
                offset += sizeof(uint64_t);
                counter_flat(i) = counter_val;
            } else {
                counter_flat(i) = i;
            }
        }
        
        tensorflow::TensorShape alg_shape({});
        tensorflow::Tensor alg_tensor(tensorflow::DT_INT32, alg_shape);
        if (offset + sizeof(int32_t) <= size) {
            int32_t alg_val;
            std::memcpy(&alg_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            alg_val = std::abs(alg_val) % 3;
            alg_tensor.scalar<int32_t>()() = alg_val;
        } else {
            alg_tensor.scalar<int32_t>()() = 1;
        }

        auto shape_op = tensorflow::ops::Const(root, shape_tensor);
        auto key_op = tensorflow::ops::Const(root, key_tensor);
        auto counter_op = tensorflow::ops::Const(root, counter_tensor);
        auto alg_op = tensorflow::ops::Const(root, alg_tensor);

        tensorflow::Node* node = nullptr;
        tensorflow::Status status = tensorflow::NodeBuilder(
                                        root.GetUniqueNameForOp("stateless_random_uniform_v2"),
                                        "StatelessRandomUniformV2")
                                        .Input(tensorflow::NodeBuilder::NodeOut(shape_op.node()))
                                        .Input(tensorflow::NodeBuilder::NodeOut(key_op.node()))
                                        .Input(tensorflow::NodeBuilder::NodeOut(counter_op.node()))
                                        .Input(tensorflow::NodeBuilder::NodeOut(alg_op.node()))
                                        .Attr("dtype", output_dtype)
                                        .Attr("Tshape", shape_tensor.dtype())
                                        .Finalize(root.graph(), &node);
        if (!status.ok()) {
            return -1;
        }

        tensorflow::Output random_op(node, 0);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({random_op}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
