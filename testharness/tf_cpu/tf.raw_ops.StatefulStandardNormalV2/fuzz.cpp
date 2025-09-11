#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/random_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include <cstring>
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
    switch (selector % 6) {  
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
            dtype = tensorflow::DT_HALF;
            break;
        case 3:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 4:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 5:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        default:
            dtype = tensorflow::DT_FLOAT;
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
        
        int64_t algorithm_value = 1;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&algorithm_value, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            algorithm_value = std::abs(algorithm_value) % 3 + 1;
        }
        
        uint8_t shape_rank = parseRank(data[offset++]);
        std::vector<int64_t> shape_dims = parseShape(data, offset, size, shape_rank);
        
        // Create a resource variable handle
        auto resource_var = tensorflow::ops::Variable(root.WithOpName("resource_var"), 
                                                     tensorflow::TensorShape({}), 
                                                     tensorflow::DT_INT64);
        
        // Create algorithm tensor
        auto algorithm_tensor = tensorflow::ops::Const(root.WithOpName("algorithm"), algorithm_value);
        
        // Create shape tensor
        tensorflow::TensorShape shape_tensor_shape({static_cast<int64_t>(shape_dims.size())});
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT64, shape_tensor_shape);
        auto shape_flat = shape_tensor.flat<int64_t>();
        for (size_t i = 0; i < shape_dims.size(); ++i) {
            shape_flat(i) = shape_dims[i];
        }
        auto shape_op = tensorflow::ops::Const(root.WithOpName("shape"), shape_tensor);
        
        // Use raw ops to create StatefulStandardNormalV2
        tensorflow::Node* result_node;
        tensorflow::NodeBuilder node_builder("stateful_standard_normal", "StatefulStandardNormalV2");
        node_builder.Input(resource_var.output.node());
        node_builder.Input(algorithm_tensor.output.node());
        node_builder.Input(shape_op.output.node());
        node_builder.Attr("dtype", output_dtype);
        
        tensorflow::Status status = root.graph()->AddNode(node_builder, &result_node);
        if (!status.ok()) {
            return -1;
        }
        
        auto result = tensorflow::Output(result_node, 0);

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
