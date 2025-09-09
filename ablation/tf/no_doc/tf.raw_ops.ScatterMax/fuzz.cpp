#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 8) {
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
            dtype = tensorflow::DT_INT64;
            break;
        case 4:
            dtype = tensorflow::DT_HALF;
            break;
        case 5:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 6:
            dtype = tensorflow::DT_UINT32;
            break;
        case 7:
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
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BFLOAT16:
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_HALF:
            fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT32:
            fillTensorWithData<uint32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT64:
            fillTensorWithData<uint64_t>(tensor, data, offset, total_size);
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
        uint8_t ref_rank = parseRank(data[offset++]);
        uint8_t indices_rank = parseRank(data[offset++]);
        uint8_t updates_rank = parseRank(data[offset++]);

        if (ref_rank == 0 || indices_rank == 0 || updates_rank == 0) {
            return 0;
        }

        std::vector<int64_t> ref_shape = parseShape(data, offset, size, ref_rank);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        std::vector<int64_t> updates_shape = parseShape(data, offset, size, updates_rank);

        if (ref_shape.empty() || indices_shape.empty() || updates_shape.empty()) {
            return 0;
        }

        tensorflow::TensorShape ref_tensor_shape(ref_shape);
        tensorflow::TensorShape indices_tensor_shape(indices_shape);
        tensorflow::TensorShape updates_tensor_shape(updates_shape);

        tensorflow::Tensor ref_tensor(dtype, ref_tensor_shape);
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, indices_tensor_shape);
        tensorflow::Tensor updates_tensor(dtype, updates_tensor_shape);

        fillTensorWithDataByType(ref_tensor, dtype, data, offset, size);
        fillTensorWithData<int32_t>(indices_tensor, data, offset, size);
        fillTensorWithDataByType(updates_tensor, dtype, data, offset, size);

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

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto ref_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto indices_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto updates_placeholder = tensorflow::ops::Placeholder(root, dtype);

        auto scatter_max = tensorflow::ops::ScatterMax(root, ref_placeholder, indices_placeholder, updates_placeholder);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;

        tensorflow::Status status = session.Run({{ref_placeholder, ref_tensor}, 
                                                 {indices_placeholder, indices_tensor}, 
                                                 {updates_placeholder, updates_tensor}}, 
                                                {scatter_max}, &outputs);

        if (!status.ok()) {
            std::cout << "ScatterMax operation failed: " << status.ToString() << std::endl;
            return 0;
        }

        std::cout << "ScatterMax operation completed successfully" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}