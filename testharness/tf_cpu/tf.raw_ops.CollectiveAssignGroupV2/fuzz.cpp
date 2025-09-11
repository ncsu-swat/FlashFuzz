#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include <iostream>
#include <cstring>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << "Error: " << message << std::endl;
    }
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
    case tensorflow::DT_INT32:
      fillTensorWithData<int32_t>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t rank1 = parseRank(data[offset++]);
        std::vector<int64_t> shape1 = parseShape(data, offset, size, rank1);
        tensorflow::TensorShape tensor_shape1(shape1);
        tensorflow::Tensor group_assignment_tensor(tensorflow::DT_INT32, tensor_shape1);
        fillTensorWithDataByType(group_assignment_tensor, tensorflow::DT_INT32, data, offset, size);

        uint8_t rank2 = parseRank(data[offset++]);
        std::vector<int64_t> shape2 = parseShape(data, offset, size, rank2);
        tensorflow::TensorShape tensor_shape2(shape2);
        tensorflow::Tensor device_index_tensor(tensorflow::DT_INT32, tensor_shape2);
        fillTensorWithDataByType(device_index_tensor, tensorflow::DT_INT32, data, offset, size);

        uint8_t rank3 = parseRank(data[offset++]);
        std::vector<int64_t> shape3 = parseShape(data, offset, size, rank3);
        tensorflow::TensorShape tensor_shape3(shape3);
        tensorflow::Tensor base_key_tensor(tensorflow::DT_INT32, tensor_shape3);
        fillTensorWithDataByType(base_key_tensor, tensorflow::DT_INT32, data, offset, size);

        auto group_assignment = tensorflow::ops::Const(root, group_assignment_tensor);
        auto device_index = tensorflow::ops::Const(root, device_index_tensor);
        auto base_key = tensorflow::ops::Const(root, base_key_tensor);

        std::cout << "Group assignment tensor shape: ";
        for (int i = 0; i < group_assignment_tensor.shape().dims(); ++i) {
            std::cout << group_assignment_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Device index tensor shape: ";
        for (int i = 0; i < device_index_tensor.shape().dims(); ++i) {
            std::cout << device_index_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Base key tensor shape: ";
        for (int i = 0; i < base_key_tensor.shape().dims(); ++i) {
            std::cout << base_key_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

        // Use raw_ops directly instead of the missing collective_ops.h
        auto collective_assign = tensorflow::ops::CollectiveAssignGroupV2(
            root.WithOpName("CollectiveAssignGroupV2"),
            group_assignment,
            device_index,
            base_key);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({collective_assign.group_size, collective_assign.group_key}, &outputs);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
