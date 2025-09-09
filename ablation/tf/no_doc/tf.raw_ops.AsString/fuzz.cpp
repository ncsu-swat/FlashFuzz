#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/str_util.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/version.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/cc/framework/scope.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

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
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 7:
            dtype = tensorflow::DT_INT64;
            break;
        case 8:
            dtype = tensorflow::DT_BOOL;
            break;
        case 9:
            dtype = tensorflow::DT_QINT8;
            break;
        case 10:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 11:
            dtype = tensorflow::DT_QINT32;
            break;
        case 12:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 13:
            dtype = tensorflow::DT_QINT16;
            break;
        case 14:
            dtype = tensorflow::DT_QUINT16;
            break;
        case 15:
            dtype = tensorflow::DT_UINT16;
            break;
        case 16:
            dtype = tensorflow::DT_COMPLEX128;
            break;
        case 17:
            dtype = tensorflow::DT_HALF;
            break;
        case 18:
            dtype = tensorflow::DT_UINT32;
            break;
        case 19:
            dtype = tensorflow::DT_UINT64;
            break;
        case 20:
            dtype = tensorflow::DT_VARIANT;
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
        case tensorflow::DT_QINT8:
            fillTensorWithData<tensorflow::qint8>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QUINT8:
            fillTensorWithData<tensorflow::quint8>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT16:
            fillTensorWithData<tensorflow::qint16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QUINT16:
            fillTensorWithData<tensorflow::quint16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_QINT32:
            fillTensorWithData<tensorflow::qint32>(tensor, data, offset, total_size);
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
        uint8_t rank = parseRank(data[offset++]);
        
        std::vector<int64_t> shape = parseShape(data, offset, size, rank);
        
        tensorflow::TensorShape tensor_shape(shape);
        tensorflow::Tensor input_tensor(input_dtype, tensor_shape);
        
        std::cout << "Input tensor dtype: " << tensorflow::DataTypeString(input_dtype) << std::endl;
        std::cout << "Input tensor shape: " << tensor_shape.DebugString() << std::endl;
        
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        
        int precision = -1;
        bool scientific = false;
        bool shortest = false;
        int width = -1;
        std::string fill = "";
        
        if (offset < size) {
            precision = static_cast<int>(data[offset++] % 10);
        }
        if (offset < size) {
            scientific = (data[offset++] % 2) == 1;
        }
        if (offset < size) {
            shortest = (data[offset++] % 2) == 1;
        }
        if (offset < size) {
            width = static_cast<int>(data[offset++] % 20);
        }
        if (offset < size) {
            fill = std::string(1, static_cast<char>(data[offset++] % 128));
        }
        
        std::cout << "Precision: " << precision << std::endl;
        std::cout << "Scientific: " << scientific << std::endl;
        std::cout << "Shortest: " << shortest << std::endl;
        std::cout << "Width: " << width << std::endl;
        std::cout << "Fill: " << fill << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, input_dtype);
        
        tensorflow::ops::AsString::Attrs attrs;
        if (precision >= 0) {
            attrs = attrs.Precision(precision);
        }
        attrs = attrs.Scientific(scientific);
        attrs = attrs.Shortest(shortest);
        if (width >= 0) {
            attrs = attrs.Width(width);
        }
        if (!fill.empty()) {
            attrs = attrs.Fill(fill);
        }
        
        auto as_string_op = tensorflow::ops::AsString(root, input_placeholder, attrs);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({{input_placeholder, input_tensor}}, {as_string_op}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "AsString operation completed successfully" << std::endl;
            std::cout << "Output tensor shape: " << outputs[0].shape().DebugString() << std::endl;
            
            auto output_flat = outputs[0].flat<tensorflow::tstring>();
            for (int i = 0; i < std::min(static_cast<int>(output_flat.size()), 10); ++i) {
                std::cout << "Output[" << i << "]: " << output_flat(i) << std::endl;
            }
        } else {
            std::cout << "AsString operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}