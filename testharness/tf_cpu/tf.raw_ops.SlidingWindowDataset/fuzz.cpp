#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/dataset_ops.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <algorithm>
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

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype; 
    switch (selector % 21) {  
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
            dtype = tensorflow::DT_STRING;
            break;
        case 7:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 8:
            dtype = tensorflow::DT_INT64;
            break;
        case 9:
            dtype = tensorflow::DT_BOOL;
            break;
        case 10:
            dtype = tensorflow::DT_QINT8;
            break;
        case 11:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 12:
            dtype = tensorflow::DT_QINT32;
            break;
        case 13:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 14:
            dtype = tensorflow::DT_QINT16;
            break;
        case 15:
            dtype = tensorflow::DT_QUINT16;
            break;
        case 16:
            dtype = tensorflow::DT_UINT16;
            break;
        case 17:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 18:
            dtype = tensorflow::DT_HALF;
            break;
        case 19:
            dtype = tensorflow::DT_UINT32;
            break;
        case 20:
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
        case tensorflow::DT_STRING: {
            auto flat = tensor.flat<tensorflow::tstring>();
            const size_t num_elements = flat.size();
            for (size_t i = 0; i < num_elements; ++i) {
                if (offset < total_size) {
                    uint8_t str_len = data[offset] % 10 + 1;
                    offset++;
                    std::string str;
                    for (uint8_t j = 0; j < str_len && offset < total_size; ++j) {
                        str += static_cast<char>(data[offset] % 128);
                        offset++;
                    }
                    flat(i) = str;
                } else {
                    flat(i) = "";
                }
            }
            break;
        }
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
        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        uint8_t input_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        // Ensure we have at least one dimension so TensorSliceDataset can slice.
        if (input_shape.empty()) {
            input_shape.push_back(1);
        }

        tensorflow::TensorShape input_tensor_shape;
        for (int64_t dim : input_shape) {
            input_tensor_shape.AddDim(dim);
        }

        tensorflow::Tensor input_tensor(input_dtype, input_tensor_shape);
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);

        int64_t dataset_stop = std::max<int64_t>(1, static_cast<int64_t>(input_tensor.NumElements()));
        auto start = tensorflow::ops::Const(root, static_cast<int64_t>(0));
        auto stop = tensorflow::ops::Const(root, dataset_stop);
        auto step = tensorflow::ops::Const(root, static_cast<int64_t>(1));

        // Build a simple range dataset as the input variant handle.
        std::vector<tensorflow::DataType> range_output_types = {tensorflow::DT_INT64};
        std::vector<tensorflow::PartialTensorShape> range_output_shapes = {tensorflow::PartialTensorShape({})};
        tensorflow::Node* range_dataset_node = nullptr;
        tensorflow::Status status = tensorflow::NodeBuilder(
                                         root.GetUniqueNameForOp("range_dataset"),
                                         "RangeDataset")
                                         .Input(start.node(), start.index())
                                         .Input(stop.node(), stop.index())
                                         .Input(step.node(), step.index())
                                         .Attr("output_types", range_output_types)
                                         .Attr("output_shapes", range_output_shapes)
                                         .Finalize(root.graph(), &range_dataset_node);
        if (!status.ok()) {
            return 0;
        }
        tensorflow::Output range_dataset(range_dataset_node);

        int64_t window_size_val = 1;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&window_size_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            window_size_val = std::abs(window_size_val) % 10 + 1;
        }
        
        int64_t window_shift_val = 1;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&window_shift_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            window_shift_val = std::abs(window_shift_val) % 10 + 1;
        }
        
        int64_t window_stride_val = 1;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&window_stride_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            window_stride_val = std::abs(window_stride_val) % 10 + 1;
        }
        
        auto window_size = tensorflow::ops::Const(root, window_size_val);
        auto window_shift = tensorflow::ops::Const(root, window_shift_val);
        auto window_stride = tensorflow::ops::Const(root, window_stride_val);

        std::vector<tensorflow::DataType> output_types = range_output_types;

        std::vector<tensorflow::PartialTensorShape> output_shapes;
        tensorflow::PartialTensorShape window_shape({-1});
        window_shape.Concatenate(range_output_shapes[0]);
        output_shapes.push_back(window_shape);
        
        bool drop_remainder = (offset < size) ? (data[offset++] % 2 == 0) : true;

        tensorflow::Node* sliding_window_node = nullptr;
        status = tensorflow::NodeBuilder(
                     root.GetUniqueNameForOp("sliding_window_dataset"),
                     "SlidingWindowDataset")
                     .Input(range_dataset.node(), range_dataset.index())
                     .Input(window_size.node(), window_size.index())
                     .Input(window_shift.node(), window_shift.index())
                     .Input(window_stride.node(), window_stride.index())
                     .Attr("drop_remainder", drop_remainder)
                     .Attr("output_types", output_types)
                     .Attr("output_shapes", output_shapes)
                     .Finalize(root.graph(), &sliding_window_node);

        if (!status.ok()) {
            return 0;
        }

        tensorflow::Output sliding_window_dataset(sliding_window_node);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({sliding_window_dataset}, &outputs);
        if (!status.ok()) {
            return 0;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
