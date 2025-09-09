#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/node_def_util.h>
#include <tensorflow/core/framework/op_def_builder.h>
#include <tensorflow/core/framework/types.pb.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseQuantizedDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 5) {
        case 0:
            dtype = tensorflow::DT_QINT8;
            break;
        case 1:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 2:
            dtype = tensorflow::DT_QINT32;
            break;
        case 3:
            dtype = tensorflow::DT_QINT16;
            break;
        case 4:
            dtype = tensorflow::DT_QUINT16;
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
    case tensorflow::DT_QINT8:
      fillTensorWithData<tensorflow::qint8>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_QUINT8:
      fillTensorWithData<tensorflow::quint8>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_QINT32:
      fillTensorWithData<tensorflow::qint32>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_QINT16:
      fillTensorWithData<tensorflow::qint16>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_QUINT16:
      fillTensorWithData<tensorflow::quint16>(tensor, data, offset, total_size);
      break;
    default:
      break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 20) {
            return 0;
        }

        tensorflow::DataType x_dtype = parseQuantizedDataType(data[offset++]);
        tensorflow::DataType y_dtype = parseQuantizedDataType(data[offset++]);
        tensorflow::DataType output_dtype = parseQuantizedDataType(data[offset++]);
        
        uint8_t x_rank = parseRank(data[offset++]);
        uint8_t y_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> x_shape = parseShape(data, offset, size, x_rank);
        std::vector<int64_t> y_shape = parseShape(data, offset, size, y_rank);
        
        tensorflow::TensorShape x_tensor_shape(x_shape);
        tensorflow::TensorShape y_tensor_shape(y_shape);
        tensorflow::TensorShape scalar_shape({});
        
        tensorflow::Tensor x_tensor(x_dtype, x_tensor_shape);
        tensorflow::Tensor y_tensor(y_dtype, y_tensor_shape);
        tensorflow::Tensor min_x_tensor(tensorflow::DT_FLOAT, scalar_shape);
        tensorflow::Tensor max_x_tensor(tensorflow::DT_FLOAT, scalar_shape);
        tensorflow::Tensor min_y_tensor(tensorflow::DT_FLOAT, scalar_shape);
        tensorflow::Tensor max_y_tensor(tensorflow::DT_FLOAT, scalar_shape);
        
        fillTensorWithDataByType(x_tensor, x_dtype, data, offset, size);
        fillTensorWithDataByType(y_tensor, y_dtype, data, offset, size);
        fillTensorWithDataByType(min_x_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(max_x_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(min_y_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(max_y_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        std::cout << "x_tensor shape: ";
        for (int i = 0; i < x_tensor.shape().dims(); ++i) {
            std::cout << x_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "y_tensor shape: ";
        for (int i = 0; i < y_tensor.shape().dims(); ++i) {
            std::cout << y_tensor.shape().dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "min_x: " << min_x_tensor.scalar<float>()() << std::endl;
        std::cout << "max_x: " << max_x_tensor.scalar<float>()() << std::endl;
        std::cout << "min_y: " << min_y_tensor.scalar<float>()() << std::endl;
        std::cout << "max_y: " << max_y_tensor.scalar<float>()() << std::endl;
        
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        node_def->set_name("quantized_mul");
        node_def->set_op("QuantizedMul");
        
        node_def->add_input("x:0");
        node_def->add_input("y:0");
        node_def->add_input("min_x:0");
        node_def->add_input("max_x:0");
        node_def->add_input("min_y:0");
        node_def->add_input("max_y:0");
        
        tensorflow::AttrValue attr_value;
        attr_value.set_type(output_dtype);
        (*node_def->mutable_attr())["Toutput"] = attr_value;
        
        tensorflow::AttrValue x_attr_value;
        x_attr_value.set_type(x_dtype);
        (*node_def->mutable_attr())["T1"] = x_attr_value;
        
        tensorflow::AttrValue y_attr_value;
        y_attr_value.set_type(y_dtype);
        (*node_def->mutable_attr())["T2"] = y_attr_value;
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            std::cout << "Failed to create session: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"x:0", x_tensor},
            {"y:0", y_tensor},
            {"min_x:0", min_x_tensor},
            {"max_x:0", max_x_tensor},
            {"min_y:0", min_y_tensor},
            {"max_y:0", max_y_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {"quantized_mul:0", "quantized_mul:1", "quantized_mul:2"};
        
        status = session->Run(inputs, output_names, {}, &outputs);
        if (!status.ok()) {
            std::cout << "Failed to run session: " << status.ToString() << std::endl;
            return 0;
        }
        
        if (outputs.size() >= 3) {
            std::cout << "Output z shape: ";
            for (int i = 0; i < outputs[0].shape().dims(); ++i) {
                std::cout << outputs[0].shape().dim_size(i) << " ";
            }
            std::cout << std::endl;
            
            std::cout << "min_z: " << outputs[1].scalar<float>()() << std::endl;
            std::cout << "max_z: " << outputs[2].scalar<float>()() << std::endl;
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}