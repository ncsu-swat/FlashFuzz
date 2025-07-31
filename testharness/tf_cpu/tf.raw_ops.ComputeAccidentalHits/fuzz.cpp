#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/candidate_sampling_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t true_classes_rank = parseRank(data[offset++]);
        std::vector<int64_t> true_classes_shape = parseShape(data, offset, size, true_classes_rank);
        
        uint8_t sampled_candidates_rank = parseRank(data[offset++]);
        std::vector<int64_t> sampled_candidates_shape = parseShape(data, offset, size, sampled_candidates_rank);
        
        if (offset + 12 > size) return 0;
        
        int32_t num_true;
        std::memcpy(&num_true, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        num_true = std::abs(num_true) % 100 + 1;
        
        int32_t seed;
        std::memcpy(&seed, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        
        int32_t seed2;
        std::memcpy(&seed2, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);

        tensorflow::TensorShape true_classes_tensor_shape(true_classes_shape);
        tensorflow::Tensor true_classes_tensor(tensorflow::DT_INT64, true_classes_tensor_shape);
        fillTensorWithDataByType(true_classes_tensor, tensorflow::DT_INT64, data, offset, size);

        tensorflow::TensorShape sampled_candidates_tensor_shape(sampled_candidates_shape);
        tensorflow::Tensor sampled_candidates_tensor(tensorflow::DT_INT64, sampled_candidates_tensor_shape);
        fillTensorWithDataByType(sampled_candidates_tensor, tensorflow::DT_INT64, data, offset, size);

        auto true_classes_input = tensorflow::ops::Const(root, true_classes_tensor);
        auto sampled_candidates_input = tensorflow::ops::Const(root, sampled_candidates_tensor);

        std::cout << "true_classes shape: ";
        for (auto dim : true_classes_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        
        std::cout << "sampled_candidates shape: ";
        for (auto dim : sampled_candidates_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        
        std::cout << "num_true: " << num_true << std::endl;
        std::cout << "seed: " << seed << std::endl;
        std::cout << "seed2: " << seed2 << std::endl;

        auto compute_accidental_hits = tensorflow::ops::ComputeAccidentalHits(
            root,
            true_classes_input,
            sampled_candidates_input,
            num_true,
            tensorflow::ops::ComputeAccidentalHits::Attrs().Seed(seed).Seed2(seed2)
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({compute_accidental_hits.indices, 
                                                compute_accidental_hits.ids, 
                                                compute_accidental_hits.weights}, &outputs);
        if (!status.ok()) {
            std::cout << "Error running session: " << status.ToString() << std::endl;
            return -1;
        }

        std::cout << "Operation completed successfully" << std::endl;
        std::cout << "Output indices shape: ";
        for (int i = 0; i < outputs[0].shape().dims(); ++i) {
            std::cout << outputs[0].shape().dim_size(i) << " ";
        }
        std::cout << std::endl;

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}