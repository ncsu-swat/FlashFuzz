#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/sparse_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <cstring>
#include <vector>
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        tensorflow::DataType data_dtype = parseDataType(data[offset++]);
        tensorflow::DataType indices_dtype = parseIndicesDataType(data[offset++]);
        tensorflow::DataType segment_ids_dtype = parseIndicesDataType(data[offset++]);
        
        uint8_t data_rank = parseRank(data[offset++]);
        std::vector<int64_t> data_shape = parseShape(data, offset, size, data_rank);
        
        if (offset >= size) return 0;
        
        uint8_t indices_size_byte = data[offset++];
        int64_t indices_size = 1 + (indices_size_byte % 10);
        
        if (data_shape.empty() || data_shape[0] < indices_size) {
            return 0;
        }
        
        uint8_t num_segments_byte = data[offset++];
        int64_t num_segments = 1 + (num_segments_byte % 5);
        
        bool sparse_gradient = (data[offset++] % 2) == 1;
        
        tensorflow::Tensor data_tensor(data_dtype, tensorflow::TensorShape(data_shape));
        fillTensorWithDataByType(data_tensor, data_dtype, data, offset, size);
        
        tensorflow::Tensor indices_tensor(indices_dtype, tensorflow::TensorShape({indices_size}));
        if (indices_dtype == tensorflow::DT_INT32) {
            auto indices_flat = indices_tensor.flat<int32_t>();
            for (int64_t i = 0; i < indices_size; ++i) {
                if (offset + sizeof(int32_t) <= size) {
                    int32_t idx;
                    std::memcpy(&idx, data + offset, sizeof(int32_t));
                    offset += sizeof(int32_t);
                    indices_flat(i) = std::abs(idx) % static_cast<int32_t>(data_shape[0]);
                } else {
                    indices_flat(i) = i % static_cast<int32_t>(data_shape[0]);
                }
            }
        } else {
            auto indices_flat = indices_tensor.flat<int64_t>();
            for (int64_t i = 0; i < indices_size; ++i) {
                if (offset + sizeof(int64_t) <= size) {
                    int64_t idx;
                    std::memcpy(&idx, data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    indices_flat(i) = std::abs(idx) % data_shape[0];
                } else {
                    indices_flat(i) = i % data_shape[0];
                }
            }
        }
        
        tensorflow::Tensor segment_ids_tensor(segment_ids_dtype, tensorflow::TensorShape({indices_size}));
        if (segment_ids_dtype == tensorflow::DT_INT32) {
            auto segment_ids_flat = segment_ids_tensor.flat<int32_t>();
            for (int64_t i = 0; i < indices_size; ++i) {
                if (offset + sizeof(int32_t) <= size) {
                    int32_t seg_id;
                    std::memcpy(&seg_id, data + offset, sizeof(int32_t));
                    offset += sizeof(int32_t);
                    segment_ids_flat(i) = std::abs(seg_id) % static_cast<int32_t>(num_segments);
                } else {
                    segment_ids_flat(i) = i % static_cast<int32_t>(num_segments);
                }
            }
        } else {
            auto segment_ids_flat = segment_ids_tensor.flat<int64_t>();
            for (int64_t i = 0; i < indices_size; ++i) {
                if (offset + sizeof(int64_t) <= size) {
                    int64_t seg_id;
                    std::memcpy(&seg_id, data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    segment_ids_flat(i) = std::abs(seg_id) % num_segments;
                } else {
                    segment_ids_flat(i) = i % num_segments;
                }
            }
        }

        auto data_input = tensorflow::ops::Const(root, data_tensor);
        auto indices_input = tensorflow::ops::Const(root, indices_tensor);
        auto segment_ids_input = tensorflow::ops::Const(root, segment_ids_tensor);

        auto sparse_segment_mean_op = tensorflow::ops::SparseSegmentMean(
            root, data_input, indices_input, segment_ids_input,
            tensorflow::ops::SparseSegmentMean::SparseGradient(sparse_gradient));

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({sparse_segment_mean_op}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
