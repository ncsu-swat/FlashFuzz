#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/candidate_sampling_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cstring>
#include <iostream>
#include <vector>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << message << std::endl;
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
        uint8_t rank = parseRank(data[offset++]);
        if (rank == 0) rank = 2;
        
        std::vector<int64_t> shape = parseShape(data, offset, size, rank);
        
        if (shape.size() < 2) {
            shape = {2, 3};
        }
        
        tensorflow::TensorShape tensor_shape;
        for (auto dim : shape) {
            tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor true_classes_tensor(tensorflow::DT_INT64, tensor_shape);
        fillTensorWithData<int64_t>(true_classes_tensor, data, offset, size);
        
        auto true_classes_flat = true_classes_tensor.flat<int64_t>();
        for (int i = 0; i < true_classes_flat.size(); ++i) {
            true_classes_flat(i) = std::abs(true_classes_flat(i)) % 1000;
        }
        
        int num_true = static_cast<int>(shape[1]);
        if (num_true < 1) num_true = 1;
        
        int num_sampled = 5;
        if (offset < size) {
            num_sampled = static_cast<int>(data[offset++] % 10) + 1;
        }
        
        bool unique = true;
        if (offset < size) {
            unique = (data[offset++] % 2) == 1;
        }
        
        int range_max = 1000;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&range_max, data + offset, sizeof(int));
            offset += sizeof(int);
            range_max = std::abs(range_max) % 10000 + 100;
        }
        
        int seed = 0;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&seed, data + offset, sizeof(int));
            offset += sizeof(int);
        }
        
        int seed2 = 0;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&seed2, data + offset, sizeof(int));
            offset += sizeof(int);
        }
        
        auto true_classes = tensorflow::ops::Const(root, true_classes_tensor);
        
        // Use raw ops to access ThreadUnsafeUnigramCandidateSampler
        auto op_attrs = tensorflow::ops::Attrs()
            .WithAttr("num_true", num_true)
            .WithAttr("num_sampled", num_sampled)
            .WithAttr("unique", unique)
            .WithAttr("range_max", range_max)
            .WithAttr("seed", seed)
            .WithAttr("seed2", seed2);
            
        auto op = tensorflow::Operation(root.WithOpName("ThreadUnsafeUnigramCandidateSampler")
            .WithAttr("num_true", num_true)
            .WithAttr("num_sampled", num_sampled)
            .WithAttr("unique", unique)
            .WithAttr("range_max", range_max)
            .WithAttr("seed", seed)
            .WithAttr("seed2", seed2));
            
        auto scope = root.WithOpName("ThreadUnsafeUnigramCandidateSampler");
        tensorflow::NodeBuilder node_builder(
            scope.GetUniqueNameForOp("ThreadUnsafeUnigramCandidateSampler"),
            "ThreadUnsafeUnigramCandidateSampler");
            
        node_builder.Input(tensorflow::ops::AsNodeOut(scope, true_classes));
        node_builder.Attr("num_true", num_true);
        node_builder.Attr("num_sampled", num_sampled);
        node_builder.Attr("unique", unique);
        node_builder.Attr("range_max", range_max);
        node_builder.Attr("seed", seed);
        node_builder.Attr("seed2", seed2);
        
        tensorflow::Node* node;
        scope.UpdateBuilder(&node_builder);
        scope.UpdateStatus(node_builder.Finalize(scope.graph(), &node));
        
        if (!scope.ok()) return 0;
        
        auto sampled_candidates = tensorflow::Output(node, 0);
        auto true_expected_count = tensorflow::Output(node, 1);
        auto sampled_expected_count = tensorflow::Output(node, 2);
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({sampled_candidates, true_expected_count, sampled_expected_count}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}