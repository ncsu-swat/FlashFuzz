#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 1
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseGradDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 2:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 3:
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
    case tensorflow::DT_BFLOAT16:
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset,
                                               total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType grad_dtype = parseGradDataType(data[offset++]);
        
        std::vector<int64_t> orig_input_shape_data;
        if (offset + 4 * sizeof(int32_t) <= size) {
            for (int i = 0; i < 4; ++i) {
                int32_t dim;
                std::memcpy(&dim, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                dim = std::abs(dim) % 10 + 1;
                orig_input_shape_data.push_back(static_cast<int64_t>(dim));
            }
        } else {
            orig_input_shape_data = {1, 4, 4, 1};
        }
        
        tensorflow::Tensor orig_input_shape_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({4}));
        auto orig_input_flat = orig_input_shape_tensor.flat<int32_t>();
        for (int i = 0; i < 4; ++i) {
            orig_input_flat(i) = static_cast<int32_t>(orig_input_shape_data[i]);
        }
        
        std::vector<int64_t> grad_shape;
        if (offset + 4 * sizeof(int32_t) <= size) {
            for (int i = 0; i < 4; ++i) {
                int32_t dim;
                std::memcpy(&dim, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                dim = std::abs(dim) % 8 + 1;
                grad_shape.push_back(static_cast<int64_t>(dim));
            }
        } else {
            grad_shape = {1, 2, 2, 1};
        }
        
        tensorflow::TensorShape grad_tensor_shape;
        for (auto dim : grad_shape) {
            grad_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor grad_tensor(grad_dtype, grad_tensor_shape);
        fillTensorWithDataByType(grad_tensor, grad_dtype, data, offset, size);
        
        std::vector<int> ksize;
        if (offset + 4 * sizeof(int32_t) <= size) {
            for (int i = 0; i < 4; ++i) {
                int32_t k;
                std::memcpy(&k, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                k = std::abs(k) % 5 + 1;
                ksize.push_back(k);
            }
        } else {
            ksize = {1, 2, 2, 1};
        }
        
        std::vector<int> strides;
        if (offset + 4 * sizeof(int32_t) <= size) {
            for (int i = 0; i < 4; ++i) {
                int32_t s;
                std::memcpy(&s, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                s = std::abs(s) % 3 + 1;
                strides.push_back(s);
            }
        } else {
            strides = {1, 1, 1, 1};
        }
        
        std::string padding = (offset < size && data[offset++] % 2 == 0) ? "SAME" : "VALID";
        std::string data_format = (offset < size && data[offset++] % 2 == 0) ? "NHWC" : "NCHW";
        
        auto orig_input_shape_op = tensorflow::ops::Const(root, orig_input_shape_tensor);
        auto grad_op = tensorflow::ops::Const(root, grad_tensor);
        
        // Use raw_ops namespace to access AvgPoolGrad
        auto avg_pool_grad = tensorflow::ops::AvgPool(
            root.WithOpName("AvgPool"),
            grad_op,
            ksize,
            strides,
            padding,
            data_format
        );
        
        // Alternative approach using raw ops
        tensorflow::NodeDef node_def;
        node_def.set_name("AvgPoolGrad");
        node_def.set_op("AvgPoolGrad");
        
        auto* attr_ksize = node_def.mutable_attr()->insert({"ksize", tensorflow::AttrValue()}).first->second.mutable_list();
        for (auto k : ksize) {
            attr_ksize->add_i(k);
        }
        
        auto* attr_strides = node_def.mutable_attr()->insert({"strides", tensorflow::AttrValue()}).first->second.mutable_list();
        for (auto s : strides) {
            attr_strides->add_i(s);
        }
        
        (*node_def.mutable_attr())["padding"].set_s(padding);
        (*node_def.mutable_attr())["data_format"].set_s(data_format);
        (*node_def.mutable_attr())["T"].set_type(grad_dtype);
        
        tensorflow::Status status;
        auto op = root.AddNode(node_def, &status);
        if (!status.ok()) {
            return -1;
        }
        
        auto avg_pool_grad_op = tensorflow::Output(op, 0);
        root.UpdateEdge(orig_input_shape_op, 0, op, 0);
        root.UpdateEdge(grad_op, 0, op, 1);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({avg_pool_grad_op}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
