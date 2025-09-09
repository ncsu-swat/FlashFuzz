#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 6;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_HALF;
            break;
        case 2:
            dtype = tensorflow::DT_BFLOAT16;
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
    try {
        size_t offset = 0;
        
        if (size < 20) return 0;
        
        tensorflow::DataType dtype = parseDataType(data[offset++]);
        
        uint8_t y_backprop_rank = parseRank(data[offset++]);
        std::vector<int64_t> y_backprop_shape = parseShape(data, offset, size, y_backprop_rank);
        
        uint8_t x_rank = parseRank(data[offset++]);
        std::vector<int64_t> x_shape = parseShape(data, offset, size, x_rank);
        
        if (x_shape.empty() || x_shape.size() < 2) {
            x_shape = {2, 3, 4, 5};
        }
        
        int64_t depth = x_shape.back();
        std::vector<int64_t> scale_shape = {depth};
        std::vector<int64_t> offset_shape = {depth};
        std::vector<int64_t> reserve_space_1_shape = {depth};
        std::vector<int64_t> reserve_space_2_shape = {depth};
        std::vector<int64_t> reserve_space_3_shape = {depth};
        
        if (y_backprop_shape.empty()) {
            y_backprop_shape = x_shape;
        }
        
        tensorflow::Tensor y_backprop_tensor(dtype, tensorflow::TensorShape(y_backprop_shape));
        tensorflow::Tensor x_tensor(dtype, tensorflow::TensorShape(x_shape));
        tensorflow::Tensor scale_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(scale_shape));
        tensorflow::Tensor offset_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(offset_shape));
        tensorflow::Tensor reserve_space_1_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(reserve_space_1_shape));
        tensorflow::Tensor reserve_space_2_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(reserve_space_2_shape));
        tensorflow::Tensor reserve_space_3_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(reserve_space_3_shape));
        
        fillTensorWithDataByType(y_backprop_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(x_tensor, dtype, data, offset, size);
        fillTensorWithData<float>(scale_tensor, data, offset, size);
        fillTensorWithData<float>(offset_tensor, data, offset, size);
        fillTensorWithData<float>(reserve_space_1_tensor, data, offset, size);
        fillTensorWithData<float>(reserve_space_2_tensor, data, offset, size);
        fillTensorWithData<float>(reserve_space_3_tensor, data, offset, size);
        
        std::cout << "y_backprop shape: ";
        for (auto dim : y_backprop_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "x shape: ";
        for (auto dim : x_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto y_backprop_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto x_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto scale_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto offset_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto reserve_space_1_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto reserve_space_2_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto reserve_space_3_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        auto fused_batch_norm_grad = tensorflow::ops::FusedBatchNormGradV3(
            root,
            y_backprop_placeholder,
            x_placeholder,
            scale_placeholder,
            reserve_space_1_placeholder,
            reserve_space_2_placeholder,
            reserve_space_3_placeholder,
            tensorflow::ops::FusedBatchNormGradV3::Attrs()
                .Epsilon(1e-5f)
                .DataFormat("NHWC")
                .IsTraining(true)
        );
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({
            {y_backprop_placeholder, y_backprop_tensor},
            {x_placeholder, x_tensor},
            {scale_placeholder, scale_tensor},
            {offset_placeholder, offset_tensor},
            {reserve_space_1_placeholder, reserve_space_1_tensor},
            {reserve_space_2_placeholder, reserve_space_2_tensor},
            {reserve_space_3_placeholder, reserve_space_3_tensor}
        }, {fused_batch_norm_grad.x_backprop, fused_batch_norm_grad.scale_backprop, fused_batch_norm_grad.offset_backprop}, {}, &outputs);
        
        if (status.ok()) {
            std::cout << "FusedBatchNormGradV3 executed successfully" << std::endl;
            std::cout << "Output tensors count: " << outputs.size() << std::endl;
        } else {
            std::cout << "FusedBatchNormGradV3 failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}