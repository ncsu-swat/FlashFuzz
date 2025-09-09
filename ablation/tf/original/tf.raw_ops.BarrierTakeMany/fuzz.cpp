#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/ops_util.h>
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
        case 11:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 12:
            dtype = tensorflow::DT_HALF;
            break;
        case 13:
            dtype = tensorflow::DT_COMPLEX64;
            break;
        case 14:
            dtype = tensorflow::DT_COMPLEX128;
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

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        tensorflow::Tensor handle_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        handle_tensor.scalar<tensorflow::tstring>()() = "test_barrier_handle";
        
        if (offset >= size) return 0;
        int32_t num_elements_val = 1 + (data[offset] % 5);
        offset++;
        
        tensorflow::Tensor num_elements_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        num_elements_tensor.scalar<int32_t>()() = num_elements_val;
        
        if (offset >= size) return 0;
        uint8_t num_component_types = 1 + (data[offset] % 3);
        offset++;
        
        std::vector<tensorflow::DataType> component_types;
        for (uint8_t i = 0; i < num_component_types; ++i) {
            if (offset >= size) break;
            tensorflow::DataType dtype = parseDataType(data[offset]);
            component_types.push_back(dtype);
            offset++;
        }
        
        if (component_types.empty()) {
            component_types.push_back(tensorflow::DT_FLOAT);
        }
        
        if (offset >= size) return 0;
        bool allow_small_batch = (data[offset] % 2) == 1;
        offset++;
        
        if (offset >= size) return 0;
        bool wait_for_incomplete = (data[offset] % 2) == 1;
        offset++;
        
        if (offset >= size) return 0;
        int timeout_ms = -1;
        if (data[offset] % 2 == 1) {
            timeout_ms = 1000 + (data[offset] % 5000);
        }
        offset++;

        std::cout << "Handle: " << handle_tensor.DebugString() << std::endl;
        std::cout << "Num elements: " << num_elements_tensor.DebugString() << std::endl;
        std::cout << "Component types count: " << component_types.size() << std::endl;
        std::cout << "Allow small batch: " << allow_small_batch << std::endl;
        std::cout << "Wait for incomplete: " << wait_for_incomplete << std::endl;
        std::cout << "Timeout ms: " << timeout_ms << std::endl;

        auto handle_op = tensorflow::ops::Const(root, handle_tensor);
        auto num_elements_op = tensorflow::ops::Const(root, num_elements_tensor);

        tensorflow::ops::BarrierTakeMany::Attrs attrs;
        attrs = attrs.AllowSmallBatch(allow_small_batch);
        attrs = attrs.WaitForIncomplete(wait_for_incomplete);
        attrs = attrs.TimeoutMs(timeout_ms);

        auto barrier_take_many = tensorflow::ops::BarrierTakeMany(
            root, handle_op, num_elements_op, component_types, attrs);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({barrier_take_many.indices, 
                                                barrier_take_many.keys}, 
                                               &outputs);
        
        if (!status.ok()) {
            std::cout << "BarrierTakeMany operation failed: " << status.ToString() << std::endl;
        } else {
            std::cout << "BarrierTakeMany operation succeeded" << std::endl;
            if (outputs.size() >= 2) {
                std::cout << "Indices output: " << outputs[0].DebugString() << std::endl;
                std::cout << "Keys output: " << outputs[1].DebugString() << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}