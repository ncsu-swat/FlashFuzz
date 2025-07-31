#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
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
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 5) {
        case 0:
            dtype = tensorflow::DT_RESOURCE;
            break;
        case 1:
            dtype = tensorflow::DT_STRING;
            break;
        case 2:
            dtype = tensorflow::DT_INT32;
            break;
        case 3:
            dtype = tensorflow::DT_INT32;
            break;
        case 4:
            dtype = tensorflow::DT_STRING;
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

void fillStringTensor(tensorflow::Tensor& tensor, const uint8_t* data,
                      size_t& offset, size_t total_size) {
    auto flat = tensor.flat<tensorflow::tstring>();
    const size_t num_elements = flat.size();

    for (size_t i = 0; i < num_elements; ++i) {
        if (offset < total_size) {
            size_t str_len = std::min(static_cast<size_t>(data[offset] % 100 + 1), total_size - offset - 1);
            offset++;
            
            if (offset + str_len <= total_size) {
                std::string str(reinterpret_cast<const char*>(data + offset), str_len);
                flat(i) = tensorflow::tstring(str);
                offset += str_len;
            } else {
                flat(i) = tensorflow::tstring("default");
                offset = total_size;
            }
        } else {
            flat(i) = tensorflow::tstring("default");
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
        case tensorflow::DT_STRING:
            fillStringTensor(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_RESOURCE:
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
        auto writer_output = tensorflow::ops::Placeholder(root.WithOpName("writer"), tensorflow::DT_RESOURCE);
        
        uint8_t logdir_rank = parseRank(data[offset++]);
        std::vector<int64_t> logdir_shape = parseShape(data, offset, size, logdir_rank);
        tensorflow::TensorShape logdir_tensor_shape(logdir_shape);
        tensorflow::Tensor logdir_tensor(tensorflow::DT_STRING, logdir_tensor_shape);
        fillStringTensor(logdir_tensor, data, offset, size);
        auto logdir_input = tensorflow::ops::Const(root.WithOpName("logdir"), logdir_tensor);

        uint8_t max_queue_rank = parseRank(data[offset++]);
        std::vector<int64_t> max_queue_shape = parseShape(data, offset, size, max_queue_rank);
        tensorflow::TensorShape max_queue_tensor_shape(max_queue_shape);
        tensorflow::Tensor max_queue_tensor(tensorflow::DT_INT32, max_queue_tensor_shape);
        fillTensorWithData<int32_t>(max_queue_tensor, data, offset, size);
        auto max_queue_input = tensorflow::ops::Const(root.WithOpName("max_queue"), max_queue_tensor);

        uint8_t flush_millis_rank = parseRank(data[offset++]);
        std::vector<int64_t> flush_millis_shape = parseShape(data, offset, size, flush_millis_rank);
        tensorflow::TensorShape flush_millis_tensor_shape(flush_millis_shape);
        tensorflow::Tensor flush_millis_tensor(tensorflow::DT_INT32, flush_millis_tensor_shape);
        fillTensorWithData<int32_t>(flush_millis_tensor, data, offset, size);
        auto flush_millis_input = tensorflow::ops::Const(root.WithOpName("flush_millis"), flush_millis_tensor);

        uint8_t filename_suffix_rank = parseRank(data[offset++]);
        std::vector<int64_t> filename_suffix_shape = parseShape(data, offset, size, filename_suffix_rank);
        tensorflow::TensorShape filename_suffix_tensor_shape(filename_suffix_shape);
        tensorflow::Tensor filename_suffix_tensor(tensorflow::DT_STRING, filename_suffix_tensor_shape);
        fillStringTensor(filename_suffix_tensor, data, offset, size);
        auto filename_suffix_input = tensorflow::ops::Const(root.WithOpName("filename_suffix"), filename_suffix_tensor);

        // Use raw_ops approach instead of summary_ops.h
        tensorflow::NodeDef node_def;
        node_def.set_name("create_summary_file_writer");
        node_def.set_op("CreateSummaryFileWriter");
        
        (*node_def.mutable_attr())["T"].set_type(tensorflow::DT_RESOURCE);
        
        tensorflow::Status status;
        auto op = root.AddNode(node_def, &status);
        
        if (!status.ok()) {
            return -1;
        }
        
        std::vector<tensorflow::Output> inputs = {
            writer_output, 
            logdir_input, 
            max_queue_input, 
            flush_millis_input, 
            filename_suffix_input
        };
        
        root.graph()->AddEdge(writer_output.node(), 0, op.node(), 0);
        root.graph()->AddEdge(logdir_input.node(), 0, op.node(), 1);
        root.graph()->AddEdge(max_queue_input.node(), 0, op.node(), 2);
        root.graph()->AddEdge(flush_millis_input.node(), 0, op.node(), 3);
        root.graph()->AddEdge(filename_suffix_input.node(), 0, op.node(), 4);

        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({}, {}, {op.name()}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}