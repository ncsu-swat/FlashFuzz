#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/public/version.h>
#include <tensorflow/cc/ops/standard_ops.h>

constexpr uint8_t MIN_RANK = 1;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 17) {
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
            dtype = tensorflow::DT_QINT8;
            break;
        case 9:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 10:
            dtype = tensorflow::DT_QINT32;
            break;
        case 11:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 12:
            dtype = tensorflow::DT_QINT16;
            break;
        case 13:
            dtype = tensorflow::DT_QUINT16;
            break;
        case 14:
            dtype = tensorflow::DT_UINT16;
            break;
        case 15:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 16:
            dtype = tensorflow::DT_HALF;
            break;
        default:
            dtype = tensorflow::DT_FLOAT;
            break;
    }
    return dtype;
}

tensorflow::DataType parseIndicesDataType(uint8_t selector) {
    return (selector % 2 == 0) ? tensorflow::DT_INT32 : tensorflow::DT_INT64;
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
            if constexpr (std::is_floating_point_v<T>) {
                if (std::isnan(value) || std::isinf(value) || value == T(0)) {
                    value = T(1);
                }
            } else if constexpr (std::is_integral_v<T>) {
                if (value == T(0)) {
                    value = T(1);
                }
            }
            flat(i) = value;
        } else {
            flat(i) = T(1);
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
        case tensorflow::DT_QINT8:
            fillTensorWithData<tensorflow::qint8>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QUINT8:
            fillTensorWithData<tensorflow::quint8>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT32:
            fillTensorWithData<tensorflow::qint32>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT16:
            fillTensorWithData<tensorflow::qint16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QUINT16:
            fillTensorWithData<tensorflow::quint16>(tensor, data, offset, total_size);
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

        tensorflow::DataType ref_dtype = parseDataType(data[offset++]);
        tensorflow::DataType indices_dtype = parseIndicesDataType(data[offset++]);
        
        uint8_t ref_rank = parseRank(data[offset++]);
        uint8_t indices_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> ref_shape = parseShape(data, offset, size, ref_rank);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        
        std::vector<int64_t> updates_shape = indices_shape;
        if (ref_rank > 1) {
            for (size_t i = 1; i < ref_shape.size(); ++i) {
                updates_shape.push_back(ref_shape[i]);
            }
        }
        
        tensorflow::TensorShape ref_tensor_shape(ref_shape);
        tensorflow::TensorShape indices_tensor_shape(indices_shape);
        tensorflow::TensorShape updates_tensor_shape(updates_shape);
        
        tensorflow::Tensor ref_tensor(ref_dtype, ref_tensor_shape);
        tensorflow::Tensor indices_tensor(indices_dtype, indices_tensor_shape);
        tensorflow::Tensor updates_tensor(ref_dtype, updates_tensor_shape);
        
        fillTensorWithDataByType(ref_tensor, ref_dtype, data, offset, size);
        
        if (indices_dtype == tensorflow::DT_INT32) {
            auto indices_flat = indices_tensor.flat<int32_t>();
            for (int i = 0; i < indices_flat.size(); ++i) {
                if (offset + sizeof(int32_t) <= size) {
                    int32_t idx;
                    std::memcpy(&idx, data + offset, sizeof(int32_t));
                    offset += sizeof(int32_t);
                    indices_flat(i) = std::abs(idx) % static_cast<int32_t>(ref_shape[0]);
                } else {
                    indices_flat(i) = 0;
                }
            }
        } else {
            auto indices_flat = indices_tensor.flat<int64_t>();
            for (int i = 0; i < indices_flat.size(); ++i) {
                if (offset + sizeof(int64_t) <= size) {
                    int64_t idx;
                    std::memcpy(&idx, data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    indices_flat(i) = std::abs(idx) % ref_shape[0];
                } else {
                    indices_flat(i) = 0;
                }
            }
        }
        
        fillTensorWithDataByType(updates_tensor, ref_dtype, data, offset, size);
        
        std::cout << "ref_tensor shape: ";
        for (int i = 0; i < ref_tensor.dims(); ++i) {
            std::cout << ref_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "indices_tensor shape: ";
        for (int i = 0; i < indices_tensor.dims(); ++i) {
            std::cout << indices_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "updates_tensor shape: ";
        for (int i = 0; i < updates_tensor.dims(); ++i) {
            std::cout << updates_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        tensorflow::GraphDef graph_def;
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto ref_placeholder = tensorflow::ops::Placeholder(root.WithOpName("ref"), ref_dtype);
        auto indices_placeholder = tensorflow::ops::Placeholder(root.WithOpName("indices"), indices_dtype);
        auto updates_placeholder = tensorflow::ops::Placeholder(root.WithOpName("updates"), ref_dtype);
        
        auto scatter_div = tensorflow::ops::ScatterDiv(root.WithOpName("scatter_div"), 
                                                      ref_placeholder, 
                                                      indices_placeholder, 
                                                      updates_placeholder);
        
        tensorflow::Status status = root.ToGraphDef(&graph_def);
        if (!status.ok()) {
            std::cout << "Failed to create graph: " << status.ToString() << std::endl;
            return 0;
        }
        
        status = session->Create(graph_def);
        if (!status.ok()) {
            std::cout << "Failed to create session: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"ref", ref_tensor},
            {"indices", indices_tensor},
            {"updates", updates_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"scatter_div"}, {}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Failed to run session: " << status.ToString() << std::endl;
        } else {
            std::cout << "ScatterDiv operation completed successfully" << std::endl;
            if (!outputs.empty()) {
                std::cout << "Output tensor shape: ";
                for (int i = 0; i < outputs[0].dims(); ++i) {
                    std::cout << outputs[0].dim_size(i) << " ";
                }
                std::cout << std::endl;
            }
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}