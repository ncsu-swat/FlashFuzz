#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/op_def_builder.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/common_runtime/device_factory.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

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
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        uint8_t gradients_rank = parseRank(data[offset++]);
        std::vector<int64_t> gradients_shape = parseShape(data, offset, size, gradients_rank);
        
        uint8_t inputs_rank = parseRank(data[offset++]);
        std::vector<int64_t> inputs_shape = parseShape(data, offset, size, inputs_rank);

        if (offset >= size) {
            return 0;
        }

        float min_val = -6.0f;
        float max_val = 6.0f;
        int num_bits = 8;
        bool narrow_range = false;

        if (offset + sizeof(float) <= size) {
            std::memcpy(&min_val, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        if (offset + sizeof(float) <= size) {
            std::memcpy(&max_val, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        if (offset + sizeof(int) <= size) {
            std::memcpy(&num_bits, data + offset, sizeof(int));
            offset += sizeof(int);
            num_bits = std::abs(num_bits % 16) + 1;
        }
        
        if (offset < size) {
            narrow_range = (data[offset++] % 2) == 1;
        }

        tensorflow::TensorShape gradients_tensor_shape(gradients_shape);
        tensorflow::TensorShape inputs_tensor_shape(inputs_shape);

        tensorflow::Tensor gradients_tensor(tensorflow::DT_FLOAT, gradients_tensor_shape);
        tensorflow::Tensor inputs_tensor(tensorflow::DT_FLOAT, inputs_tensor_shape);

        fillTensorWithDataByType(gradients_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(inputs_tensor, tensorflow::DT_FLOAT, data, offset, size);

        std::cout << "Gradients tensor shape: ";
        for (int i = 0; i < gradients_tensor.dims(); ++i) {
            std::cout << gradients_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Inputs tensor shape: ";
        for (int i = 0; i < inputs_tensor.dims(); ++i) {
            std::cout << inputs_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "min: " << min_val << ", max: " << max_val << ", num_bits: " << num_bits << ", narrow_range: " << narrow_range << std::endl;

        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));

        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        node_def->set_name("fake_quant_grad");
        node_def->set_op("FakeQuantWithMinMaxArgsGradient");
        
        tensorflow::NodeDef::AttrValue min_attr;
        min_attr.set_f(min_val);
        (*node_def->mutable_attr())["min"] = min_attr;
        
        tensorflow::NodeDef::AttrValue max_attr;
        max_attr.set_f(max_val);
        (*node_def->mutable_attr())["max"] = max_attr;
        
        tensorflow::NodeDef::AttrValue num_bits_attr;
        num_bits_attr.set_i(num_bits);
        (*node_def->mutable_attr())["num_bits"] = num_bits_attr;
        
        tensorflow::NodeDef::AttrValue narrow_range_attr;
        narrow_range_attr.set_b(narrow_range);
        (*node_def->mutable_attr())["narrow_range"] = narrow_range_attr;

        node_def->add_input("gradients:0");
        node_def->add_input("inputs:0");

        tensorflow::NodeDef* gradients_placeholder = graph_def.add_node();
        gradients_placeholder->set_name("gradients");
        gradients_placeholder->set_op("Placeholder");
        tensorflow::NodeDef::AttrValue gradients_dtype_attr;
        gradients_dtype_attr.set_type(tensorflow::DT_FLOAT);
        (*gradients_placeholder->mutable_attr())["dtype"] = gradients_dtype_attr;

        tensorflow::NodeDef* inputs_placeholder = graph_def.add_node();
        inputs_placeholder->set_name("inputs");
        inputs_placeholder->set_op("Placeholder");
        tensorflow::NodeDef::AttrValue inputs_dtype_attr;
        inputs_dtype_attr.set_type(tensorflow::DT_FLOAT);
        (*inputs_placeholder->mutable_attr())["dtype"] = inputs_dtype_attr;

        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            std::cout << "Failed to create session: " << status.ToString() << std::endl;
            return 0;
        }

        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"gradients:0", gradients_tensor},
            {"inputs:0", inputs_tensor}
        };

        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"fake_quant_grad:0"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "Operation executed successfully. Output shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
        }

        session->Close();

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}