#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>

#define MAX_RANK 6
#define MIN_RANK 3
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseInputDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 1:
            dtype = tensorflow::DT_COMPLEX128;
            break;
    }
    return dtype;
}

tensorflow::DataType parseOutputDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
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
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType input_dtype = parseInputDataType(data[offset++]);
        tensorflow::DataType output_dtype = parseOutputDataType(data[offset++]);
        
        uint8_t rank = parseRank(data[offset++]);
        std::vector<int64_t> shape = parseShape(data, offset, size, rank);
        
        if (shape.size() < 3) {
            return 0;
        }
        
        shape[shape.size()-1] = shape[shape.size()-1] / 2 + 1;
        
        tensorflow::TensorShape input_shape(shape);
        tensorflow::Tensor input_tensor(input_dtype, input_shape);
        
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        
        std::vector<int32_t> fft_length_data(3);
        for (int i = 0; i < 3; ++i) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t val;
                std::memcpy(&val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                fft_length_data[i] = std::abs(val) % 20 + 1;
            } else {
                fft_length_data[i] = static_cast<int32_t>(shape[shape.size()-3+i]);
            }
        }
        
        tensorflow::TensorShape fft_length_shape({3});
        tensorflow::Tensor fft_length_tensor(tensorflow::DT_INT32, fft_length_shape);
        auto fft_length_flat = fft_length_tensor.flat<int32_t>();
        for (int i = 0; i < 3; ++i) {
            fft_length_flat(i) = fft_length_data[i];
        }
        
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto fft_length_op = tensorflow::ops::Const(root, fft_length_tensor);
        
        // Use raw_ops namespace for IRFFT3D
        tensorflow::Output irfft3d_op;
        
        tensorflow::NodeDef node_def;
        node_def.set_op("IRFFT3D");
        node_def.set_name(root.UniqueName("IRFFT3D"));
        
        // Add inputs to NodeDef
        tensorflow::NodeDefBuilder node_def_builder(node_def.name(), node_def.op());
        node_def_builder.Input(input_op.node()->name(), 0, input_dtype);
        node_def_builder.Input(fft_length_op.node()->name(), 0, tensorflow::DT_INT32);
        
        // Add attributes
        node_def_builder.Attr("Tcomplex", input_dtype);
        node_def_builder.Attr("Treal", output_dtype);
        
        tensorflow::Status status = node_def_builder.Finalize(&node_def);
        if (!status.ok()) {
            return -1;
        }
        
        tensorflow::Node* node;
        status = root.graph()->AddNode(node_def, &node);
        if (!status.ok()) {
            return -1;
        }
        
        // Add edges
        root.graph()->AddEdge(input_op.node(), 0, node, 0);
        root.graph()->AddEdge(fft_length_op.node(), 0, node, 1);
        
        irfft3d_op = tensorflow::Output(node, 0);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        status = session.Run({irfft3d_op}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}