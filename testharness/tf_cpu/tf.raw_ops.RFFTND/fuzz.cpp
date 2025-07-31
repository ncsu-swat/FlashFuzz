#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 1
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
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
    }
    return dtype;
}

tensorflow::DataType parseComplexDataType(uint8_t selector) {
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
        tensorflow::DataType complex_dtype = parseComplexDataType(data[offset++]);
        
        uint8_t input_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        tensorflow::TensorShape input_tensor_shape;
        for (int64_t dim : input_shape) {
            input_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor input_tensor(input_dtype, input_tensor_shape);
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        
        uint8_t fft_length_size = (offset < size) ? (data[offset++] % input_rank) + 1 : input_rank;
        if (fft_length_size > input_rank) fft_length_size = input_rank;
        
        std::vector<int32_t> fft_length_data;
        for (uint8_t i = 0; i < fft_length_size; ++i) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t val;
                std::memcpy(&val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                val = std::abs(val) % 20 + 1;
                fft_length_data.push_back(val);
            } else {
                fft_length_data.push_back(input_shape[i % input_rank]);
            }
        }
        
        tensorflow::TensorShape fft_length_shape;
        fft_length_shape.AddDim(fft_length_size);
        tensorflow::Tensor fft_length_tensor(tensorflow::DT_INT32, fft_length_shape);
        auto fft_length_flat = fft_length_tensor.flat<int32_t>();
        for (size_t i = 0; i < fft_length_data.size(); ++i) {
            fft_length_flat(i) = fft_length_data[i];
        }
        
        std::vector<int32_t> axes_data;
        for (uint8_t i = 0; i < fft_length_size; ++i) {
            if (offset < size) {
                int32_t axis = static_cast<int32_t>(data[offset++] % input_rank);
                axes_data.push_back(axis);
            } else {
                axes_data.push_back(i % input_rank);
            }
        }
        
        tensorflow::TensorShape axes_shape;
        axes_shape.AddDim(fft_length_size);
        tensorflow::Tensor axes_tensor(tensorflow::DT_INT32, axes_shape);
        auto axes_flat = axes_tensor.flat<int32_t>();
        for (size_t i = 0; i < axes_data.size(); ++i) {
            axes_flat(i) = axes_data[i];
        }
        
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto fft_length_op = tensorflow::ops::Const(root, fft_length_tensor);
        auto axes_op = tensorflow::ops::Const(root, axes_tensor);
        
        // Use raw_ops namespace for RFFTND
        auto rfftnd_op = tensorflow::ops::internal::RawOp(
            root.WithOpName("RFFTND"),
            "RFFTND",
            {input_op.node(), fft_length_op.node(), axes_op.node()},
            {{"Treal", input_dtype}, {"Tcomplex", complex_dtype}});
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({rfftnd_op}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}