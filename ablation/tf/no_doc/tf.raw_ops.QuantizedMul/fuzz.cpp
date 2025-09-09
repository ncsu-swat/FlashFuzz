#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/core/framework/node_def_util.h>
#include <tensorflow/core/kernels/quantized_ops_utils.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 3) {
        case 0:
            dtype = tensorflow::DT_QINT8;
            break;
        case 1:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 2:
            dtype = tensorflow::DT_QINT32;
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
    case tensorflow::DT_QINT8:
      fillTensorWithData<tensorflow::qint8>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_QUINT8:
      fillTensorWithData<tensorflow::quint8>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_QINT32:
      fillTensorWithData<tensorflow::qint32>(tensor, data, offset, total_size);
      break;
    case tensorflow::DT_FLOAT:
      fillTensorWithData<float>(tensor, data, offset, total_size);
      break;
    default:
      return;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType x_dtype = parseDataType(data[offset++]);
        tensorflow::DataType y_dtype = parseDataType(data[offset++]);
        
        uint8_t x_rank = parseRank(data[offset++]);
        uint8_t y_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> x_shape = parseShape(data, offset, size, x_rank);
        std::vector<int64_t> y_shape = parseShape(data, offset, size, y_rank);
        
        tensorflow::TensorShape x_tensor_shape(x_shape);
        tensorflow::TensorShape y_tensor_shape(y_shape);
        
        tensorflow::Tensor x_tensor(x_dtype, x_tensor_shape);
        tensorflow::Tensor y_tensor(y_dtype, y_tensor_shape);
        
        fillTensorWithDataByType(x_tensor, x_dtype, data, offset, size);
        fillTensorWithDataByType(y_tensor, y_dtype, data, offset, size);
        
        float min_x = -1.0f;
        float max_x = 1.0f;
        float min_y = -1.0f;
        float max_y = 1.0f;
        
        if (offset + sizeof(float) <= size) {
            std::memcpy(&min_x, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        if (offset + sizeof(float) <= size) {
            std::memcpy(&max_x, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        if (offset + sizeof(float) <= size) {
            std::memcpy(&min_y, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        if (offset + sizeof(float) <= size) {
            std::memcpy(&max_y, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        tensorflow::Tensor min_x_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_x_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor min_y_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_y_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        
        min_x_tensor.scalar<float>()() = min_x;
        max_x_tensor.scalar<float>()() = max_x;
        min_y_tensor.scalar<float>()() = min_y;
        max_y_tensor.scalar<float>()() = max_y;
        
        std::cout << "X tensor shape: ";
        for (int i = 0; i < x_tensor_shape.dims(); ++i) {
            std::cout << x_tensor_shape.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Y tensor shape: ";
        for (int i = 0; i < y_tensor_shape.dims(); ++i) {
            std::cout << y_tensor_shape.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "X dtype: " << tensorflow::DataTypeString(x_dtype) << std::endl;
        std::cout << "Y dtype: " << tensorflow::DataTypeString(y_dtype) << std::endl;
        std::cout << "min_x: " << min_x << ", max_x: " << max_x << std::endl;
        std::cout << "min_y: " << min_y << ", max_y: " << max_y << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto x_placeholder = tensorflow::ops::Placeholder(root.WithOpName("x"), x_dtype);
        auto y_placeholder = tensorflow::ops::Placeholder(root.WithOpName("y"), y_dtype);
        auto min_x_placeholder = tensorflow::ops::Placeholder(root.WithOpName("min_x"), tensorflow::DT_FLOAT);
        auto max_x_placeholder = tensorflow::ops::Placeholder(root.WithOpName("max_x"), tensorflow::DT_FLOAT);
        auto min_y_placeholder = tensorflow::ops::Placeholder(root.WithOpName("min_y"), tensorflow::DT_FLOAT);
        auto max_y_placeholder = tensorflow::ops::Placeholder(root.WithOpName("max_y"), tensorflow::DT_FLOAT);
        
        tensorflow::NodeDef node_def;
        node_def.set_name("quantized_mul");
        node_def.set_op("QuantizedMul");
        node_def.add_input("x");
        node_def.add_input("y");
        node_def.add_input("min_x");
        node_def.add_input("max_x");
        node_def.add_input("min_y");
        node_def.add_input("max_y");
        
        tensorflow::AttrValue t1_attr;
        t1_attr.set_type(x_dtype);
        node_def.mutable_attr()->insert({"T1", t1_attr});
        
        tensorflow::AttrValue t2_attr;
        t2_attr.set_type(y_dtype);
        node_def.mutable_attr()->insert({"T2", t2_attr});
        
        tensorflow::AttrValue toutput_attr;
        toutput_attr.set_type(tensorflow::DT_QINT32);
        node_def.mutable_attr()->insert({"Toutput", toutput_attr});
        
        tensorflow::GraphDef graph_def;
        root.ToGraphDef(&graph_def);
        
        auto new_node = graph_def.add_node();
        *new_node = node_def;
        
        tensorflow::SessionOptions session_options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            std::cout << "Failed to create session: " << status.ToString() << std::endl;
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"x", x_tensor},
            {"y", y_tensor},
            {"min_x", min_x_tensor},
            {"max_x", max_x_tensor},
            {"min_y", min_y_tensor},
            {"max_y", max_y_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {"quantized_mul:0", "quantized_mul:1", "quantized_mul:2"};
        
        status = session->Run(inputs, output_names, {}, &outputs);
        if (!status.ok()) {
            std::cout << "Failed to run session: " << status.ToString() << std::endl;
        } else {
            std::cout << "QuantizedMul operation completed successfully" << std::endl;
            if (outputs.size() >= 3) {
                std::cout << "Output tensor shape: ";
                for (int i = 0; i < outputs[0].shape().dims(); ++i) {
                    std::cout << outputs[0].shape().dim_size(i) << " ";
                }
                std::cout << std::endl;
                std::cout << "Output min: " << outputs[1].scalar<float>()() << std::endl;
                std::cout << "Output max: " << outputs[2].scalar<float>()() << std::endl;
            }
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}