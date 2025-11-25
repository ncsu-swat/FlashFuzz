#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <algorithm>
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
    switch (selector % 13) {
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
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 8:
            dtype = tensorflow::DT_UINT16;
            break;
        case 9:
            dtype = tensorflow::DT_HALF;
            break;
        case 10:
            dtype = tensorflow::DT_UINT32;
            break;
        case 11:
            dtype = tensorflow::DT_UINT64;
            break;
        case 12:
            dtype = tensorflow::DT_BOOL;
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
        default:
            break;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        auto readString = [&](size_t max_len, const std::string& fallback) {
            if (offset >= size) {
                return fallback;
            }
            const uint8_t len = data[offset++] % max_len;
            const size_t available = size - offset;
            const size_t take = std::min<size_t>(len, available);
            std::string result(reinterpret_cast<const char*>(data + offset), take);
            offset += take;
            if (result.empty()) {
                return fallback;
            }
            return result;
        };

        const std::string shared_name = readString(16, "writer");
        const std::string container = readString(16, "container");

        tensorflow::Node* writer_node = nullptr;
        tensorflow::NodeBuilder writer_builder("summary_writer", "SummaryWriter");
        writer_builder.Attr("shared_name", shared_name);
        writer_builder.Attr("container", container);
        tensorflow::Status writer_status = writer_builder.Finalize(root.graph(), &writer_node);
        root.UpdateStatus(writer_status);
        if (!writer_status.ok() || writer_node == nullptr) {
            return 0;
        }
        tensorflow::Output writer(writer_node, 0);
        
        int64_t step_value = 0;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&step_value, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        tensorflow::Tensor step_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        step_tensor.scalar<int64_t>()() = step_value;
        tensorflow::Output step = tensorflow::ops::Const(root, step_tensor);

        std::string tag_value = "histogram_tag";
        if (offset + 1 <= size) {
            uint8_t tag_len = data[offset] % 20 + 1;
            offset++;
            tag_value = "";
            for (uint8_t i = 0; i < tag_len && offset < size; ++i) {
                tag_value += static_cast<char>(data[offset] % 26 + 'a');
                offset++;
            }
        }
        tensorflow::Tensor tag_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        tag_tensor.scalar<tensorflow::tstring>()() = tag_value;
        tensorflow::Output tag = tensorflow::ops::Const(root, tag_tensor);

        if (offset >= size) return 0;
        
        tensorflow::DataType values_dtype = parseDataType(data[offset]);
        offset++;
        
        if (offset >= size) return 0;
        uint8_t values_rank = parseRank(data[offset]);
        offset++;
        
        std::vector<int64_t> values_shape = parseShape(data, offset, size, values_rank);
        
        tensorflow::TensorShape values_tensor_shape;
        for (int64_t dim : values_shape) {
            values_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor values_tensor(values_dtype, values_tensor_shape);
        fillTensorWithDataByType(values_tensor, values_dtype, data, offset, size);
        
        tensorflow::Output values = tensorflow::ops::Const(root, values_tensor);

        tensorflow::Node* write_node = nullptr;
        tensorflow::NodeBuilder write_builder("write_histogram_summary", "WriteHistogramSummary");
        write_builder.Input(tensorflow::NodeBuilder::NodeOut(writer.node()));
        write_builder.Input(tensorflow::NodeBuilder::NodeOut(step.node()));
        write_builder.Input(tensorflow::NodeBuilder::NodeOut(tag.node()));
        write_builder.Input(tensorflow::NodeBuilder::NodeOut(values.node()));
        write_builder.Attr("T", values_dtype);
        tensorflow::Status write_status = write_builder.Finalize(root.graph(), &write_node);
        root.UpdateStatus(write_status);
        (void)write_node;
        
    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
