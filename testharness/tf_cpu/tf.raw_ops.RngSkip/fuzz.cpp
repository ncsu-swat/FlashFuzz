#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_handle.h"
#include <cstring>
#include <iostream>
#include <vector>

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
    switch (selector % 2) {  
        case 0:
            dtype = tensorflow::DT_INT64;
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
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        default:
            return;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t algorithm_rank = parseRank(data[offset++]);
        std::vector<int64_t> algorithm_shape = parseShape(data, offset, size, algorithm_rank);
        tensorflow::TensorShape algorithm_tensor_shape(algorithm_shape);
        tensorflow::Tensor algorithm_tensor(tensorflow::DT_INT64, algorithm_tensor_shape);
        fillTensorWithDataByType(algorithm_tensor, tensorflow::DT_INT64, data, offset, size);
        
        uint8_t delta_rank = parseRank(data[offset++]);
        std::vector<int64_t> delta_shape = parseShape(data, offset, size, delta_rank);
        tensorflow::TensorShape delta_tensor_shape(delta_shape);
        tensorflow::Tensor delta_tensor(tensorflow::DT_INT64, delta_tensor_shape);
        fillTensorWithDataByType(delta_tensor, tensorflow::DT_INT64, data, offset, size);

        tensorflow::ResourceHandle resource_handle;
        resource_handle.set_device("/cpu:0");
        resource_handle.set_container("test_container");
        resource_handle.set_name("test_rng_resource");
        resource_handle.set_hash_code(12345);
        resource_handle.set_maybe_type_name("AnonymousRandomRNGStateGenerator");
        
        tensorflow::Tensor resource_tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
        resource_tensor.scalar<tensorflow::ResourceHandle>()() = resource_handle;

        auto algorithm_input = tensorflow::ops::Const(root, algorithm_tensor);
        auto delta_input = tensorflow::ops::Const(root, delta_tensor);
        auto resource_input = tensorflow::ops::Const(root, resource_tensor);

        // Use raw_ops.RngSkip through the NodeBuilder API
        tensorflow::NodeBuilder node_builder("RngSkip", "RngSkip");
        node_builder.Input(resource_input.node());
        node_builder.Input(algorithm_input.node());
        node_builder.Input(delta_input.node());
        
        tensorflow::Node* rng_skip_node;
        tensorflow::Status status = root.graph()->AddNode(node_builder, &rng_skip_node);
        if (!status.ok()) {
            return -1;
        }
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({tensorflow::Output(rng_skip_node, 0)}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}