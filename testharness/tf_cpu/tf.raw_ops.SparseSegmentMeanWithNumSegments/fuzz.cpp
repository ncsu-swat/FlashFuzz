#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/sparse_ops.h"
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
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 4) {
        case 0:
            dtype = tensorflow::DT_BFLOAT16;
            break;
        case 1:
            dtype = tensorflow::DT_HALF;
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

tensorflow::DataType parseIndicesDataType(uint8_t selector) {
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
    if (size < 20) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType data_dtype = parseDataType(data[offset++]);
        tensorflow::DataType indices_dtype = parseIndicesDataType(data[offset++]);
        tensorflow::DataType segment_ids_dtype = parseIndicesDataType(data[offset++]);
        tensorflow::DataType num_segments_dtype = parseIndicesDataType(data[offset++]);
        
        uint8_t data_rank = parseRank(data[offset++]);
        std::vector<int64_t> data_shape = parseShape(data, offset, size, data_rank);
        
        if (offset >= size) return 0;
        
        uint8_t indices_size_byte = data[offset++];
        int64_t indices_size = 1 + (indices_size_byte % 10);
        
        if (offset >= size) return 0;
        
        uint8_t num_segments_byte = data[offset++];
        int64_t num_segments_val = 1 + (num_segments_byte % 10);
        
        tensorflow::Tensor data_tensor(data_dtype, tensorflow::TensorShape(data_shape));
        fillTensorWithDataByType(data_tensor, data_dtype, data, offset, size);
        
        tensorflow::Tensor indices_tensor(indices_dtype, tensorflow::TensorShape({indices_size}));
        fillTensorWithDataByType(indices_tensor, indices_dtype, data, offset, size);
        
        tensorflow::Tensor segment_ids_tensor(segment_ids_dtype, tensorflow::TensorShape({indices_size}));
        fillTensorWithDataByType(segment_ids_tensor, segment_ids_dtype, data, offset, size);
        
        tensorflow::Tensor num_segments_tensor(num_segments_dtype, tensorflow::TensorShape({}));
        if (num_segments_dtype == tensorflow::DT_INT32) {
            num_segments_tensor.scalar<int32_t>()() = static_cast<int32_t>(num_segments_val);
        } else {
            num_segments_tensor.scalar<int64_t>()() = num_segments_val;
        }
        
        if (indices_dtype == tensorflow::DT_INT32) {
            auto indices_flat = indices_tensor.flat<int32_t>();
            for (int i = 0; i < indices_size; ++i) {
                indices_flat(i) = std::abs(indices_flat(i)) % static_cast<int32_t>(data_tensor.shape().dim_size(0));
            }
        } else {
            auto indices_flat = indices_tensor.flat<int64_t>();
            for (int i = 0; i < indices_size; ++i) {
                indices_flat(i) = std::abs(indices_flat(i)) % data_tensor.shape().dim_size(0);
            }
        }
        
        if (segment_ids_dtype == tensorflow::DT_INT32) {
            auto segment_ids_flat = segment_ids_tensor.flat<int32_t>();
            for (int i = 0; i < indices_size; ++i) {
                segment_ids_flat(i) = std::abs(segment_ids_flat(i)) % static_cast<int32_t>(num_segments_val);
            }
        } else {
            auto segment_ids_flat = segment_ids_tensor.flat<int64_t>();
            for (int i = 0; i < indices_size; ++i) {
                segment_ids_flat(i) = std::abs(segment_ids_flat(i)) % num_segments_val;
            }
        }
        
        auto data_input = tensorflow::ops::Placeholder(root, data_dtype);
        auto indices_input = tensorflow::ops::Placeholder(root, indices_dtype);
        auto segment_ids_input = tensorflow::ops::Placeholder(root, segment_ids_dtype);
        auto num_segments_input = tensorflow::ops::Placeholder(root, num_segments_dtype);
        
        bool sparse_gradient = (data[0] % 2) == 1;
        
        auto sparse_segment_mean = tensorflow::ops::SparseSegmentMeanWithNumSegments(
            root, data_input, indices_input, segment_ids_input, num_segments_input,
            tensorflow::ops::SparseSegmentMeanWithNumSegments::Attrs().SparseGradient(sparse_gradient));
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{data_input, data_tensor}, 
             {indices_input, indices_tensor}, 
             {segment_ids_input, segment_ids_tensor},
             {num_segments_input, num_segments_tensor}},
            {sparse_segment_mean}, &outputs);
            
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}