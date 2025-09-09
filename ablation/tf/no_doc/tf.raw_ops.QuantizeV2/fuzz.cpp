#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/array_ops.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

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
            dtype = tensorflow::DT_INT32;
            break;
        case 3:
            dtype = tensorflow::DT_UINT8;
            break;
        case 4:
            dtype = tensorflow::DT_INT8;
            break;
        case 5:
            dtype = tensorflow::DT_HALF;
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
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        uint8_t input_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        tensorflow::TensorShape tensor_shape(input_shape);
        tensorflow::Tensor input_tensor(input_dtype, tensor_shape);
        
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        
        float min_range = 0.0f;
        float max_range = 1.0f;
        
        if (offset + sizeof(float) <= size) {
            std::memcpy(&min_range, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        if (offset + sizeof(float) <= size) {
            std::memcpy(&max_range, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        if (min_range > max_range) {
            std::swap(min_range, max_range);
        }
        
        tensorflow::DataType T = tensorflow::DT_QUINT8;
        if (offset < size) {
            uint8_t type_selector = data[offset++];
            switch (type_selector % 3) {
                case 0:
                    T = tensorflow::DT_QINT8;
                    break;
                case 1:
                    T = tensorflow::DT_QUINT8;
                    break;
                case 2:
                    T = tensorflow::DT_QINT32;
                    break;
            }
        }
        
        std::string mode = "MIN_COMBINED";
        if (offset < size) {
            uint8_t mode_selector = data[offset++];
            switch (mode_selector % 4) {
                case 0:
                    mode = "MIN_COMBINED";
                    break;
                case 1:
                    mode = "MIN_FIRST";
                    break;
                case 2:
                    mode = "SCALED";
                    break;
                case 3:
                    mode = "UNIT_SCALE";
                    break;
            }
        }
        
        bool round_mode = false;
        if (offset < size) {
            round_mode = (data[offset++] % 2) == 1;
        }
        
        bool narrow_range = false;
        if (offset < size) {
            narrow_range = (data[offset++] % 2) == 1;
        }
        
        int axis = -1;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&axis, data + offset, sizeof(int));
            offset += sizeof(int);
            axis = axis % (input_rank + 1);
            if (axis < 0) axis = -1;
        }
        
        std::cout << "Input tensor shape: ";
        for (int64_t dim : input_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        std::cout << "Input dtype: " << tensorflow::DataTypeString(input_dtype) << std::endl;
        std::cout << "Output dtype: " << tensorflow::DataTypeString(T) << std::endl;
        std::cout << "Min range: " << min_range << std::endl;
        std::cout << "Max range: " << max_range << std::endl;
        std::cout << "Mode: " << mode << std::endl;
        std::cout << "Round mode: " << round_mode << std::endl;
        std::cout << "Narrow range: " << narrow_range << std::endl;
        std::cout << "Axis: " << axis << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, input_dtype, 
            tensorflow::ops::Placeholder::Shape(tensor_shape));
        auto min_range_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto max_range_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        tensorflow::Node* quantize_node;
        tensorflow::NodeBuilder builder("quantize_v2", "QuantizeV2");
        builder.Input(input_placeholder.node())
               .Input(min_range_placeholder.node())
               .Input(max_range_placeholder.node())
               .Attr("T", T)
               .Attr("mode", mode)
               .Attr("round_mode", round_mode)
               .Attr("narrow_range", narrow_range);
        
        if (axis != -1) {
            builder.Attr("axis", axis);
        }
        
        tensorflow::Status status = builder.Finalize(root.graph(), &quantize_node);
        if (!status.ok()) {
            std::cout << "Failed to create QuantizeV2 node: " << status.ToString() << std::endl;
            return 0;
        }
        
        tensorflow::ClientSession session(root);
        
        tensorflow::Tensor min_range_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        min_range_tensor.scalar<float>()() = min_range;
        
        tensorflow::Tensor max_range_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        max_range_tensor.scalar<float>()() = max_range;
        
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({{input_placeholder, input_tensor},
                             {min_range_placeholder, min_range_tensor},
                             {max_range_placeholder, max_range_tensor}},
                            {tensorflow::Output(quantize_node, 0),
                             tensorflow::Output(quantize_node, 1),
                             tensorflow::Output(quantize_node, 2)},
                            &outputs);
        
        if (status.ok()) {
            std::cout << "QuantizeV2 operation completed successfully" << std::endl;
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].shape().dims(); ++i) {
                std::cout << outputs[0].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "QuantizeV2 operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}