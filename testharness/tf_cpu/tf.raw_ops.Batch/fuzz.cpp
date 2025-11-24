#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/graph/node_builder.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10
#define MAX_NUM_TENSORS 5

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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_tensors_byte = data[offset++];
        int num_tensors = (num_tensors_byte % MAX_NUM_TENSORS) + 1;
        
        uint8_t num_batch_threads_byte = data[offset++];
        int num_batch_threads = (num_batch_threads_byte % 8) + 1;
        
        uint8_t max_batch_size_byte = data[offset++];
        int max_batch_size = (max_batch_size_byte % 32) + 1;
        
        uint32_t batch_timeout_micros;
        if (offset + sizeof(batch_timeout_micros) <= size) {
            std::memcpy(&batch_timeout_micros, data + offset, sizeof(batch_timeout_micros));
            offset += sizeof(batch_timeout_micros);
            batch_timeout_micros = (batch_timeout_micros % 1000000) + 1000;
        } else {
            batch_timeout_micros = 10000;
        }
        
        uint32_t grad_timeout_micros;
        if (offset + sizeof(grad_timeout_micros) <= size) {
            std::memcpy(&grad_timeout_micros, data + offset, sizeof(grad_timeout_micros));
            offset += sizeof(grad_timeout_micros);
            grad_timeout_micros = (grad_timeout_micros % 1000000) + 1000;
        } else {
            grad_timeout_micros = 10000;
        }

        tensorflow::DataType dtype = parseDataType(data[offset++]);
        uint8_t rank = parseRank(data[offset++]);
        std::vector<int64_t> shape = parseShape(data, offset, size, rank);

        std::vector<tensorflow::Output> input_tensors;
        
        for (int i = 0; i < num_tensors; ++i) {
            auto placeholder = tensorflow::ops::Placeholder(root, dtype);
            input_tensors.push_back(placeholder);
        }

        std::vector<tensorflow::DataType> in_types(num_tensors, dtype);
        std::vector<tensorflow::NodeBuilder::NodeOut> input_nodes;
        input_nodes.reserve(input_tensors.size());
        for (const auto& input : input_tensors) {
            input_nodes.push_back(tensorflow::ops::AsNodeOut(root, input));
        }

        tensorflow::Node* batch_node = nullptr;
        auto builder = tensorflow::NodeBuilder(root.GetUniqueNameForOp("Batch"), "Batch")
                           .Input(input_nodes)
                           .Attr("num_batch_threads", num_batch_threads)
                           .Attr("max_batch_size", max_batch_size)
                           .Attr("batch_timeout_micros", static_cast<int64_t>(batch_timeout_micros))
                           .Attr("grad_timeout_micros", static_cast<int64_t>(grad_timeout_micros))
                           .Attr("T", in_types);
        root.UpdateStatus(builder.Finalize(root.graph(), &batch_node));
        
        if (!root.ok() || batch_node == nullptr) {
            return -1;
        }
        
        // Get the outputs
        std::vector<tensorflow::Output> batched_tensors;
        for (int i = 0; i < num_tensors; ++i) {
            batched_tensors.push_back(tensorflow::Output(batch_node, i));
        }
        auto batch_index = tensorflow::Output(batch_node, num_tensors);
        auto id = tensorflow::Output(batch_node, num_tensors + 1);

        tensorflow::ClientSession session(root);
        tensorflow::Status status;
        
        tensorflow::ClientSession::FeedType feed_dict;
        for (int i = 0; i < num_tensors; ++i) {
            tensorflow::Tensor tensor(dtype, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            feed_dict.emplace(input_tensors[i], tensor);
        }

        std::vector<tensorflow::Tensor> outputs;
        if (num_tensors > 0) {
            status = session.Run(feed_dict, {batched_tensors[0], batch_index, id}, &outputs);
        } else {
            status = session.Run(feed_dict, {batch_index, id}, &outputs);
        }
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
