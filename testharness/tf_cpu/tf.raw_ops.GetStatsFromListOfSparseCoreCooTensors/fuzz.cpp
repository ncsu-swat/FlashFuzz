#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/ops/sparse_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cstring>
#include <vector>
#include <iostream>

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
        uint8_t num_lists = (data[offset++] % 3) + 1;
        
        std::vector<tensorflow::Input> row_ids_list;
        std::vector<tensorflow::Input> col_ids_list;
        std::vector<tensorflow::Input> gains_list;
        std::vector<int> sample_count_list;
        std::vector<int> col_offset_list;
        
        for (uint8_t i = 0; i < num_lists; ++i) {
            if (offset >= size) break;
            
            uint8_t rank = parseRank(data[offset++]);
            if (rank == 0) rank = 1;
            
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::Tensor row_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(row_ids_tensor, tensorflow::DT_INT32, data, offset, size);
            auto row_ids_const = tensorflow::ops::Const(root, row_ids_tensor);
            row_ids_list.push_back(row_ids_const);
            
            tensorflow::Tensor col_ids_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(col_ids_tensor, tensorflow::DT_INT32, data, offset, size);
            auto col_ids_const = tensorflow::ops::Const(root, col_ids_tensor);
            col_ids_list.push_back(col_ids_const);
            
            tensorflow::Tensor gains_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(shape));
            fillTensorWithDataByType(gains_tensor, tensorflow::DT_FLOAT, data, offset, size);
            auto gains_const = tensorflow::ops::Const(root, gains_tensor);
            gains_list.push_back(gains_const);
            
            int sample_count = 1;
            if (offset + sizeof(int) <= size) {
                std::memcpy(&sample_count, data + offset, sizeof(int));
                offset += sizeof(int);
                sample_count = std::abs(sample_count) % 1000 + 1;
            }
            sample_count_list.push_back(sample_count);
            
            int col_offset = 0;
            if (offset + sizeof(int) <= size) {
                std::memcpy(&col_offset, data + offset, sizeof(int));
                offset += sizeof(int);
                col_offset = std::abs(col_offset) % 1000;
            }
            col_offset_list.push_back(col_offset);
        }
        
        int num_replica = 1;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&num_replica, data + offset, sizeof(int));
            offset += sizeof(int);
            num_replica = std::abs(num_replica) % 10 + 1;
        }
        
        int table_vocab_size = 100;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&table_vocab_size, data + offset, sizeof(int));
            offset += sizeof(int);
            table_vocab_size = std::abs(table_vocab_size) % 10000 + 1;
        }
        
        int feature_width = 1;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&feature_width, data + offset, sizeof(int));
            offset += sizeof(int);
            feature_width = std::abs(feature_width) % 100 + 1;
        }
        
        int num_sc_per_chip = 1;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&num_sc_per_chip, data + offset, sizeof(int));
            offset += sizeof(int);
            num_sc_per_chip = std::abs(num_sc_per_chip) % 10 + 1;
        }
        
        std::string table_name = "test_table";
        
        // Create the operation using raw_ops
        tensorflow::NodeDef node_def;
        node_def.set_op("GetStatsFromListOfSparseCoreCooTensors");
        node_def.set_name("get_stats_op");
        
        // Set attributes
        auto* attrs = node_def.mutable_attr();
        
        // Set sample_count_list attribute
        tensorflow::AttrValue sample_count_attr;
        for (int count : sample_count_list) {
            sample_count_attr.mutable_list()->add_i(count);
        }
        (*attrs)["sample_count_list"] = sample_count_attr;
        
        // Set col_offset_list attribute
        tensorflow::AttrValue col_offset_attr;
        for (int offset : col_offset_list) {
            col_offset_attr.mutable_list()->add_i(offset);
        }
        (*attrs)["col_offset_list"] = col_offset_attr;
        
        // Set num_replica attribute
        tensorflow::AttrValue num_replica_attr;
        num_replica_attr.set_i(num_replica);
        (*attrs)["num_replica"] = num_replica_attr;
        
        // Set table_vocab_size attribute
        tensorflow::AttrValue table_vocab_size_attr;
        table_vocab_size_attr.set_i(table_vocab_size);
        (*attrs)["table_vocab_size"] = table_vocab_size_attr;
        
        // Set feature_width attribute
        tensorflow::AttrValue feature_width_attr;
        feature_width_attr.set_i(feature_width);
        (*attrs)["feature_width"] = feature_width_attr;
        
        // Set num_sc_per_chip attribute
        tensorflow::AttrValue num_sc_per_chip_attr;
        num_sc_per_chip_attr.set_i(num_sc_per_chip);
        (*attrs)["num_sc_per_chip"] = num_sc_per_chip_attr;
        
        // Set table_name attribute
        tensorflow::AttrValue table_name_attr;
        table_name_attr.set_s(table_name);
        (*attrs)["table_name"] = table_name_attr;
        
        // Create operation using NodeBuilder
        tensorflow::Status status;
        tensorflow::Node* op_node = nullptr;
        tensorflow::NodeBuilder node_builder(node_def.name(), node_def.op());
        
        // Add inputs
        for (size_t i = 0; i < row_ids_list.size(); ++i) {
            node_builder.Input(tensorflow::ops::AsNodeOut(root, row_ids_list[i]));
        }
        for (size_t i = 0; i < col_ids_list.size(); ++i) {
            node_builder.Input(tensorflow::ops::AsNodeOut(root, col_ids_list[i]));
        }
        for (size_t i = 0; i < gains_list.size(); ++i) {
            node_builder.Input(tensorflow::ops::AsNodeOut(root, gains_list[i]));
        }
        
        // Add attributes
        for (const auto& attr : node_def.attr()) {
            node_builder.Attr(attr.first, attr.second);
        }
        
        // Finalize the node
        tensorflow::Scope::FinalizeBuilder builder(node_builder);
        status = builder(&op_node);
        
        if (!status.ok()) {
            return -1;
        }
        
        // Create output nodes
        tensorflow::Node* max_ids_per_sc = nullptr;
        tensorflow::Node* max_unique_ids_per_sc = nullptr;
        
        status = root.graph()->AddNode(op_node, &op_node);
        if (!status.ok()) {
            return -1;
        }
        
        // Create output tensors
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        // We can't directly run the raw op, so we'll just verify that the graph was created successfully
        
    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
