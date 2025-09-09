#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/training_ops.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/cc/client/client_session.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
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
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    default:
      return;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType dtype = parseDataType(data[offset++]);
        uint8_t rank = parseRank(data[offset++]);
        
        std::vector<int64_t> shape = parseShape(data, offset, size, rank);
        tensorflow::TensorShape tensor_shape(shape);
        
        tensorflow::Tensor var_tensor(dtype, tensor_shape);
        tensorflow::Tensor accum_tensor(dtype, tensor_shape);
        tensorflow::Tensor linear_tensor(dtype, tensor_shape);
        tensorflow::Tensor grad_tensor(dtype, tensor_shape);
        
        tensorflow::Tensor lr_tensor(dtype, tensorflow::TensorShape({}));
        tensorflow::Tensor l1_tensor(dtype, tensorflow::TensorShape({}));
        tensorflow::Tensor l2_tensor(dtype, tensorflow::TensorShape({}));
        tensorflow::Tensor l2_shrinkage_tensor(dtype, tensorflow::TensorShape({}));
        tensorflow::Tensor lr_power_tensor(dtype, tensorflow::TensorShape({}));
        
        fillTensorWithDataByType(var_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(accum_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(linear_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(grad_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(lr_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(l1_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(l2_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(l2_shrinkage_tensor, dtype, data, offset, size);
        fillTensorWithDataByType(lr_power_tensor, dtype, data, offset, size);
        
        std::cout << "var shape: ";
        for (int i = 0; i < var_tensor.shape().dims(); ++i) {
            std::cout << var_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto var_placeholder = tensorflow::ops::Placeholder(root.WithOpName("var"), dtype);
        auto accum_placeholder = tensorflow::ops::Placeholder(root.WithOpName("accum"), dtype);
        auto linear_placeholder = tensorflow::ops::Placeholder(root.WithOpName("linear"), dtype);
        auto grad_placeholder = tensorflow::ops::Placeholder(root.WithOpName("grad"), dtype);
        auto lr_placeholder = tensorflow::ops::Placeholder(root.WithOpName("lr"), dtype);
        auto l1_placeholder = tensorflow::ops::Placeholder(root.WithOpName("l1"), dtype);
        auto l2_placeholder = tensorflow::ops::Placeholder(root.WithOpName("l2"), dtype);
        auto l2_shrinkage_placeholder = tensorflow::ops::Placeholder(root.WithOpName("l2_shrinkage"), dtype);
        auto lr_power_placeholder = tensorflow::ops::Placeholder(root.WithOpName("lr_power"), dtype);
        
        auto apply_ftrl_v2 = tensorflow::ops::ApplyFtrlV2(
            root.WithOpName("apply_ftrl_v2"),
            var_placeholder,
            accum_placeholder,
            linear_placeholder,
            grad_placeholder,
            lr_placeholder,
            l1_placeholder,
            l2_placeholder,
            l2_shrinkage_placeholder,
            lr_power_placeholder
        );
        
        tensorflow::GraphDef graph;
        TF_CHECK_OK(root.ToGraphDef(&graph));
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        TF_CHECK_OK(session->Create(graph));
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"var", var_tensor},
            {"accum", accum_tensor},
            {"linear", linear_tensor},
            {"grad", grad_tensor},
            {"lr", lr_tensor},
            {"l1", l1_tensor},
            {"l2", l2_tensor},
            {"l2_shrinkage", l2_shrinkage_tensor},
            {"lr_power", lr_power_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session->Run(inputs, {"apply_ftrl_v2"}, {}, &outputs);
        
        if (status.ok()) {
            std::cout << "ApplyFtrlV2 operation completed successfully" << std::endl;
        } else {
            std::cout << "ApplyFtrlV2 operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}