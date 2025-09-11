#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include <cstring>
#include <vector>
#include <iostream>

#define MAX_RANK 4
#define MIN_RANK 1
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t num_features = (data[offset++] % 5) + 1;
        
        std::vector<tensorflow::Input> float_values;
        std::vector<tensorflow::Input> bucket_boundaries;
        
        for (uint8_t i = 0; i < num_features; ++i) {
            if (offset >= size) break;
            
            uint8_t rank = parseRank(data[offset++]);
            if (rank == 0) rank = 1;
            
            std::vector<int64_t> float_shape = parseShape(data, offset, size, rank);
            
            tensorflow::Tensor float_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(float_shape));
            fillTensorWithData<float>(float_tensor, data, offset, size);
            
            auto float_input = tensorflow::ops::Const(root, float_tensor);
            float_values.push_back(float_input);
            
            if (offset >= size) break;
            
            uint8_t boundary_rank = parseRank(data[offset++]);
            if (boundary_rank == 0) boundary_rank = 1;
            
            std::vector<int64_t> boundary_shape = parseShape(data, offset, size, boundary_rank);
            
            tensorflow::Tensor boundary_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(boundary_shape));
            fillTensorWithData<float>(boundary_tensor, data, offset, size);
            
            auto boundary_input = tensorflow::ops::Const(root, boundary_tensor);
            bucket_boundaries.push_back(boundary_input);
        }
        
        if (float_values.empty() || bucket_boundaries.empty()) {
            return 0;
        }
        
        // Use raw_ops API instead of the missing boosted_trees_ops.h
        auto bucketize_op = tensorflow::ops::Operation(
            root.WithOpName("BoostedTreesBucketize"),
            "BoostedTreesBucketize",
            float_values,
            bucket_boundaries);
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({bucketize_op.output(0)}, &outputs);
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
