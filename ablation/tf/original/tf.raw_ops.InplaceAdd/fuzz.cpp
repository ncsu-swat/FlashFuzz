#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/core/framework/node_def.pb.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/strcat.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session_options.h>

using namespace tensorflow;

constexpr uint8_t MIN_RANK = 1;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype; 
    switch (selector % 15) {  
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
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 7:
            dtype = tensorflow::DT_INT64;
            break;
        case 8:
            dtype = tensorflow::DT_BOOL;
            break;
        case 9:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 10:
            dtype = tensorflow::DT_UINT16;
            break;
        case 11:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 12:
            dtype = tensorflow::DT_HALF;
            break;
        case 13:
            dtype = tensorflow::DT_UINT32;
            break;
        case 14:
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

        tensorflow::DataType dtype = parseDataType(data[offset++]);
        uint8_t x_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> x_shape = parseShape(data, offset, size, x_rank);
        
        if (x_shape.empty() || x_shape[0] <= 0) {
            return 0;
        }
        
        tensorflow::TensorShape x_tensor_shape(x_shape);
        tensorflow::Tensor x_tensor(dtype, x_tensor_shape);
        
        fillTensorWithDataByType(x_tensor, dtype, data, offset, size);
        
        if (offset >= size) {
            return 0;
        }
        
        uint8_t i_size_byte = data[offset++];
        int32_t i_size = (i_size_byte % 5) + 1;
        if (i_size > x_shape[0]) {
            i_size = x_shape[0];
        }
        
        tensorflow::TensorShape i_tensor_shape({i_size});
        tensorflow::Tensor i_tensor(tensorflow::DT_INT32, i_tensor_shape);
        
        auto i_flat = i_tensor.flat<int32_t>();
        for (int32_t idx = 0; idx < i_size; ++idx) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t i_val;
                std::memcpy(&i_val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                i_val = std::abs(i_val) % x_shape[0];
                i_flat(idx) = i_val;
            } else {
                i_flat(idx) = idx % x_shape[0];
            }
        }
        
        std::vector<int64_t> v_shape = x_shape;
        v_shape[0] = i_size;
        tensorflow::TensorShape v_tensor_shape(v_shape);
        tensorflow::Tensor v_tensor(dtype, v_tensor_shape);
        
        fillTensorWithDataByType(v_tensor, dtype, data, offset, size);
        
        std::cout << "x tensor shape: ";
        for (int i = 0; i < x_tensor.dims(); ++i) {
            std::cout << x_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "i tensor shape: ";
        for (int i = 0; i < i_tensor.dims(); ++i) {
            std::cout << i_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "v tensor shape: ";
        for (int i = 0; i < v_tensor.dims(); ++i) {
            std::cout << v_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto x_placeholder = tensorflow::ops::Placeholder(root.WithOpName("x"), dtype);
        auto i_placeholder = tensorflow::ops::Placeholder(root.WithOpName("i"), tensorflow::DT_INT32);
        auto v_placeholder = tensorflow::ops::Placeholder(root.WithOpName("v"), dtype);
        
        tensorflow::NodeDef node_def;
        node_def.set_name("inplace_add");
        node_def.set_op("InplaceAdd");
        node_def.add_input("x");
        node_def.add_input("i");
        node_def.add_input("v");
        (*node_def.mutable_attr())["T"].set_type(dtype);
        
        tensorflow::GraphDef graph_def;
        TF_RETURN_IF_ERROR(root.ToGraphDef(&graph_def));
        
        auto new_node = graph_def.add_node();
        *new_node = node_def;
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        TF_RETURN_IF_ERROR(session->Create(graph_def));
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"x", x_tensor},
            {"i", i_tensor},
            {"v", v_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session->Run(inputs, {"inplace_add"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "InplaceAdd operation completed successfully" << std::endl;
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "InplaceAdd operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}