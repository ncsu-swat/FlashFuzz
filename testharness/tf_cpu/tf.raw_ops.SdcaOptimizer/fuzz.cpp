#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/training_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
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
      fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset,
                                               total_size);
      break;
    case tensorflow::DT_HALF:
      fillTensorWithData<Eigen::half>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_COMPLEX64:
      fillTensorWithData<tensorflow::complex64>(tensor, data, offset,
                                                total_size);
      break;
    case tensorflow::DT_COMPLEX128:
      fillTensorWithData<tensorflow::complex128>(tensor, data, offset,
                                                 total_size);
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
    if (size < 100) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_sparse_groups = (data[offset++] % 3) + 1;
        uint8_t num_dense_groups = (data[offset++] % 3) + 1;
        uint8_t num_examples = (data[offset++] % 5) + 1;

        std::vector<tensorflow::Output> sparse_example_indices;
        std::vector<tensorflow::Output> sparse_feature_indices;
        std::vector<tensorflow::Output> sparse_feature_values;
        std::vector<tensorflow::Output> sparse_indices;
        std::vector<tensorflow::Output> sparse_weights;
        
        for (uint8_t i = 0; i < num_sparse_groups; ++i) {
            uint8_t rank1 = parseRank(data[offset++]);
            std::vector<int64_t> shape1 = parseShape(data, offset, size, rank1);
            tensorflow::Tensor tensor1(tensorflow::DT_INT64, tensorflow::TensorShape(shape1));
            fillTensorWithDataByType(tensor1, tensorflow::DT_INT64, data, offset, size);
            sparse_example_indices.push_back(tensorflow::ops::Const(root, tensor1));

            uint8_t rank2 = parseRank(data[offset++]);
            std::vector<int64_t> shape2 = parseShape(data, offset, size, rank2);
            tensorflow::Tensor tensor2(tensorflow::DT_INT64, tensorflow::TensorShape(shape2));
            fillTensorWithDataByType(tensor2, tensorflow::DT_INT64, data, offset, size);
            sparse_feature_indices.push_back(tensorflow::ops::Const(root, tensor2));

            uint8_t rank3 = parseRank(data[offset++]);
            std::vector<int64_t> shape3 = parseShape(data, offset, size, rank3);
            tensorflow::Tensor tensor3(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape3));
            fillTensorWithDataByType(tensor3, tensorflow::DT_FLOAT, data, offset, size);
            sparse_feature_values.push_back(tensorflow::ops::Const(root, tensor3));

            uint8_t rank4 = parseRank(data[offset++]);
            std::vector<int64_t> shape4 = parseShape(data, offset, size, rank4);
            tensorflow::Tensor tensor4(tensorflow::DT_INT64, tensorflow::TensorShape(shape4));
            fillTensorWithDataByType(tensor4, tensorflow::DT_INT64, data, offset, size);
            sparse_indices.push_back(tensorflow::ops::Const(root, tensor4));

            uint8_t rank5 = parseRank(data[offset++]);
            std::vector<int64_t> shape5 = parseShape(data, offset, size, rank5);
            tensorflow::Tensor tensor5(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape5));
            fillTensorWithDataByType(tensor5, tensorflow::DT_FLOAT, data, offset, size);
            sparse_weights.push_back(tensorflow::ops::Const(root, tensor5));
        }

        std::vector<tensorflow::Output> dense_features;
        std::vector<tensorflow::Output> dense_weights;
        
        for (uint8_t i = 0; i < num_dense_groups; ++i) {
            uint8_t rank6 = parseRank(data[offset++]);
            std::vector<int64_t> shape6 = parseShape(data, offset, size, rank6);
            tensorflow::Tensor tensor6(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape6));
            fillTensorWithDataByType(tensor6, tensorflow::DT_FLOAT, data, offset, size);
            dense_features.push_back(tensorflow::ops::Const(root, tensor6));

            uint8_t rank7 = parseRank(data[offset++]);
            std::vector<int64_t> shape7 = parseShape(data, offset, size, rank7);
            tensorflow::Tensor tensor7(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape7));
            fillTensorWithDataByType(tensor7, tensorflow::DT_FLOAT, data, offset, size);
            dense_weights.push_back(tensorflow::ops::Const(root, tensor7));
        }

        tensorflow::Tensor example_weights_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({num_examples}));
        fillTensorWithDataByType(example_weights_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto example_weights = tensorflow::ops::Const(root, example_weights_tensor);

        tensorflow::Tensor example_labels_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({num_examples}));
        fillTensorWithDataByType(example_labels_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto example_labels = tensorflow::ops::Const(root, example_labels_tensor);

        tensorflow::Tensor example_state_data_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({num_examples, 4}));
        fillTensorWithDataByType(example_state_data_tensor, tensorflow::DT_FLOAT, data, offset, size);
        auto example_state_data = tensorflow::ops::Const(root, example_state_data_tensor);

        std::vector<std::string> loss_types = {"logistic_loss", "squared_loss", "hinge_loss", "smooth_hinge_loss", "poisson_loss"};
        std::string loss_type = loss_types[data[offset++] % loss_types.size()];

        float l1_val = 0.01f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&l1_val, data + offset, sizeof(float));
            offset += sizeof(float);
            l1_val = std::abs(l1_val);
            if (l1_val > 1.0f) l1_val = 1.0f;
        }

        float l2_val = 0.01f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&l2_val, data + offset, sizeof(float));
            offset += sizeof(float);
            l2_val = std::abs(l2_val);
            if (l2_val > 1.0f) l2_val = 1.0f;
        }

        int64_t num_loss_partitions = (data[offset++] % 4) + 1;
        int64_t num_inner_iterations = (data[offset++] % 10) + 1;
        bool adaptative = (data[offset++] % 2) == 1;

        auto sdca_optimizer = tensorflow::ops::Raw::SdcaOptimizer(
            root,
            sparse_example_indices,
            sparse_feature_indices,
            sparse_feature_values,
            dense_features,
            example_weights,
            example_labels,
            sparse_indices,
            sparse_weights,
            dense_weights,
            example_state_data,
            loss_type,
            l1_val,
            l2_val,
            num_loss_partitions,
            num_inner_iterations,
            adaptative
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({sdca_optimizer.out_example_state_data}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
