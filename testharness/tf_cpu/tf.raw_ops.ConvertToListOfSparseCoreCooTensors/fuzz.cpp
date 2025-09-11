#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/sparse_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>
#include <cstring>
#include <vector>
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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t indices_rank = parseRank(data[offset++]);
        std::vector<int64_t> indices_shape = parseShape(data, offset, size, indices_rank);
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(indices_shape));
        fillTensorWithDataByType(indices_tensor, tensorflow::DT_INT32, data, offset, size);
        
        uint8_t values_rank = parseRank(data[offset++]);
        std::vector<int64_t> values_shape = parseShape(data, offset, size, values_rank);
        tensorflow::Tensor values_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(values_shape));
        fillTensorWithDataByType(values_tensor, tensorflow::DT_INT32, data, offset, size);
        
        uint8_t weights_rank = parseRank(data[offset++]);
        std::vector<int64_t> weights_shape = parseShape(data, offset, size, weights_rank);
        tensorflow::Tensor weights_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(weights_shape));
        fillTensorWithDataByType(weights_tensor, tensorflow::DT_FLOAT, data, offset, size);

        int sample_count = 1;
        int num_sc_per_chip = 1;
        int row_offset = 0;
        int col_offset = 0;
        int col_shift = 0;
        int num_sc_shards = 1;
        int stacked_table_sample_count = 1;
        std::string combiner = "sum";

        if (offset + sizeof(int) <= size) {
            std::memcpy(&sample_count, data + offset, sizeof(int));
            offset += sizeof(int);
            sample_count = std::max(1, std::abs(sample_count) % 100 + 1);
        }

        if (offset + sizeof(int) <= size) {
            std::memcpy(&num_sc_per_chip, data + offset, sizeof(int));
            offset += sizeof(int);
            num_sc_per_chip = std::max(1, std::abs(num_sc_per_chip) % 10 + 1);
        }

        if (offset + sizeof(int) <= size) {
            std::memcpy(&row_offset, data + offset, sizeof(int));
            offset += sizeof(int);
            row_offset = std::max(0, std::abs(row_offset) % 100);
        }

        if (offset + sizeof(int) <= size) {
            std::memcpy(&col_offset, data + offset, sizeof(int));
            offset += sizeof(int);
            col_offset = std::max(0, std::abs(col_offset) % 100);
        }

        if (offset + sizeof(int) <= size) {
            std::memcpy(&col_shift, data + offset, sizeof(int));
            offset += sizeof(int);
            col_shift = std::max(0, std::abs(col_shift) % 100);
        }

        if (offset + sizeof(int) <= size) {
            std::memcpy(&num_sc_shards, data + offset, sizeof(int));
            offset += sizeof(int);
            num_sc_shards = std::max(1, std::abs(num_sc_shards) % 10 + 1);
        }

        if (offset + sizeof(int) <= size) {
            std::memcpy(&stacked_table_sample_count, data + offset, sizeof(int));
            offset += sizeof(int);
            stacked_table_sample_count = std::max(1, std::abs(stacked_table_sample_count) % 100 + 1);
        }

        if (offset < size) {
            uint8_t combiner_selector = data[offset++];
            switch (combiner_selector % 3) {
                case 0: combiner = "sum"; break;
                case 1: combiner = "mean"; break;
                case 2: combiner = "sqrtn"; break;
            }
        }

        auto indices_input = tensorflow::ops::Const(root, indices_tensor);
        auto values_input = tensorflow::ops::Const(root, values_tensor);
        auto weights_input = tensorflow::ops::Const(root, weights_tensor);

        std::cout << "Indices shape: ";
        for (auto dim : indices_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "Values shape: ";
        for (auto dim : values_shape) std::cout << dim << " ";
        std::cout << std::endl;
        
        std::cout << "Weights shape: ";
        for (auto dim : weights_shape) std::cout << dim << " ";
        std::cout << std::endl;

        std::cout << "sample_count: " << sample_count << std::endl;
        std::cout << "num_sc_per_chip: " << num_sc_per_chip << std::endl;
        std::cout << "row_offset: " << row_offset << std::endl;
        std::cout << "col_offset: " << col_offset << std::endl;
        std::cout << "col_shift: " << col_shift << std::endl;
        std::cout << "num_sc_shards: " << num_sc_shards << std::endl;
        std::cout << "stacked_table_sample_count: " << stacked_table_sample_count << std::endl;
        std::cout << "combiner: " << combiner << std::endl;

        // Use raw_ops directly since ConvertToListOfSparseCoreCooTensors is not in the standard ops
        tensorflow::OutputList row_ids_list;
        tensorflow::OutputList col_ids_list;
        tensorflow::OutputList gains_list;

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        // Create a NodeDef for the raw op
        tensorflow::NodeDef node_def;
        node_def.set_name("ConvertToListOfSparseCoreCooTensors");
        node_def.set_op("ConvertToListOfSparseCoreCooTensors");
        
        // Add inputs
        tensorflow::AddNodeInput("indices", indices_input.node()->name(), 0, &node_def);
        tensorflow::AddNodeInput("values", values_input.node()->name(), 0, &node_def);
        tensorflow::AddNodeInput("weights", weights_input.node()->name(), 0, &node_def);
        
        // Add attributes
        tensorflow::AddNodeAttr("sample_count", sample_count, &node_def);
        tensorflow::AddNodeAttr("num_sc_per_chip", num_sc_per_chip, &node_def);
        tensorflow::AddNodeAttr("row_offset", row_offset, &node_def);
        tensorflow::AddNodeAttr("col_offset", col_offset, &node_def);
        tensorflow::AddNodeAttr("col_shift", col_shift, &node_def);
        tensorflow::AddNodeAttr("num_sc_shards", num_sc_shards, &node_def);
        tensorflow::AddNodeAttr("stacked_table_sample_count", stacked_table_sample_count, &node_def);
        tensorflow::AddNodeAttr("combiner", combiner, &node_def);
        
        // Create the operation
        tensorflow::Status status;
        auto op = root.AddOperation(node_def, &status);
        
        if (!status.ok()) {
            std::cout << "Error creating operation: " << status.ToString() << std::endl;
            return -1;
        }
        
        // Run the session
        status = session.Run({op.output(0), op.output(1), op.output(2)}, &outputs);
        
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
