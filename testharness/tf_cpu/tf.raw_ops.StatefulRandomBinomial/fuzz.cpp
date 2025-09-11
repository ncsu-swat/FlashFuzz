#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/random_ops.h"
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
    std::cerr << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 5) {
        case 0:
            dtype = tensorflow::DT_HALF;
            break;
        case 1:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 2:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 3:
            dtype = tensorflow::DT_INT32;
            break;
        case 4:
            dtype = tensorflow::DT_INT64;
            break;
    }
    return dtype;
}

tensorflow::DataType parseShapeDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 2) {
        case 0:
            dtype = tensorflow::DT_INT32;
            break;
        case 1:
            dtype = tensorflow::DT_INT64;
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
        auto resource_var = tensorflow::ops::ResourceVariable(
            root.WithOpName("resource_var"), tensorflow::TensorShape({}), tensorflow::DT_INT64);

        tensorflow::Tensor algorithm_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        if (offset + sizeof(int64_t) <= size) {
            int64_t alg_val;
            std::memcpy(&alg_val, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            alg_val = std::abs(alg_val) % 3;
            algorithm_tensor.scalar<int64_t>()() = alg_val;
        } else {
            algorithm_tensor.scalar<int64_t>()() = 0;
        }
        auto algorithm = tensorflow::ops::Const(root.WithOpName("algorithm"), algorithm_tensor);

        uint8_t shape_rank = parseRank(data[offset % size]);
        offset++;
        std::vector<int64_t> shape_dims = parseShape(data, offset, size, shape_rank);
        tensorflow::DataType shape_dtype = parseShapeDataType(data[offset % size]);
        offset++;

        tensorflow::Tensor shape_tensor;
        if (shape_dtype == tensorflow::DT_INT32) {
            shape_tensor = tensorflow::Tensor(tensorflow::DT_INT32, tensorflow::TensorShape({static_cast<int64_t>(shape_dims.size())}));
            auto shape_flat = shape_tensor.flat<int32_t>();
            for (size_t i = 0; i < shape_dims.size(); ++i) {
                shape_flat(i) = static_cast<int32_t>(shape_dims[i]);
            }
        } else {
            shape_tensor = tensorflow::Tensor(tensorflow::DT_INT64, tensorflow::TensorShape({static_cast<int64_t>(shape_dims.size())}));
            auto shape_flat = shape_tensor.flat<int64_t>();
            for (size_t i = 0; i < shape_dims.size(); ++i) {
                shape_flat(i) = shape_dims[i];
            }
        }
        auto shape = tensorflow::ops::Const(root.WithOpName("shape"), shape_tensor);

        uint8_t counts_rank = parseRank(data[offset % size]);
        offset++;
        std::vector<int64_t> counts_shape = parseShape(data, offset, size, counts_rank);
        tensorflow::DataType counts_dtype = parseDataType(data[offset % size]);
        offset++;

        tensorflow::Tensor counts_tensor(counts_dtype, tensorflow::TensorShape(counts_shape));
        fillTensorWithDataByType(counts_tensor, counts_dtype, data, offset, size);
        auto counts = tensorflow::ops::Const(root.WithOpName("counts"), counts_tensor);

        uint8_t probs_rank = parseRank(data[offset % size]);
        offset++;
        std::vector<int64_t> probs_shape = parseShape(data, offset, size, probs_rank);

        tensorflow::Tensor probs_tensor(counts_dtype, tensorflow::TensorShape(probs_shape));
        fillTensorWithDataByType(probs_tensor, counts_dtype, data, offset, size);
        auto probs = tensorflow::ops::Const(root.WithOpName("probs"), probs_tensor);

        tensorflow::DataType output_dtype = parseDataType(data[offset % size]);
        offset++;

        tensorflow::Tensor init_value(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        init_value.scalar<int64_t>()() = 12345;
        auto init_op = tensorflow::ops::Assign(
            root.WithOpName("init_var"), resource_var, 
            tensorflow::ops::Const(root, init_value));

        // Create the StatefulRandomBinomial op using raw_ops
        std::vector<tensorflow::Output> result_outputs;
        tensorflow::Status status = tensorflow::ops::internal::StatefulRandomBinomial(
            root.WithOpName("stateful_random_binomial"),
            resource_var, algorithm, shape, counts, probs,
            output_dtype, &result_outputs);
        
        if (!status.ok()) {
            return -1;
        }
        
        auto result = result_outputs[0];

        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> init_outputs;
        tensorflow::Status init_status = session.Run({init_op}, &init_outputs);
        if (!init_status.ok()) {
            return -1;
        }

        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({result}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
