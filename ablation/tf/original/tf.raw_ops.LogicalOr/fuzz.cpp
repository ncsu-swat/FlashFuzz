#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/public/session.h>

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
    case tensorflow::DT_BOOL:
      fillTensorWithData<bool>(tensor, data, offset, total_size);
      break;
    default:
      return;
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
        
        if (size < 3) {
            return 0;
        }

        uint8_t rank_x = parseRank(data[offset++]);
        uint8_t rank_y = parseRank(data[offset++]);

        std::vector<int64_t> shape_x = parseShape(data, offset, size, rank_x);
        std::vector<int64_t> shape_y = parseShape(data, offset, size, rank_y);

        tensorflow::TensorShape tensor_shape_x(shape_x);
        tensorflow::TensorShape tensor_shape_y(shape_y);

        tensorflow::Tensor tensor_x(tensorflow::DT_BOOL, tensor_shape_x);
        tensorflow::Tensor tensor_y(tensorflow::DT_BOOL, tensor_shape_y);

        fillTensorWithDataByType(tensor_x, tensorflow::DT_BOOL, data, offset, size);
        fillTensorWithDataByType(tensor_y, tensorflow::DT_BOOL, data, offset, size);

        std::cout << "Tensor X shape: ";
        for (int i = 0; i < tensor_x.dims(); ++i) {
            std::cout << tensor_x.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Tensor Y shape: ";
        for (int i = 0; i < tensor_y.dims(); ++i) {
            std::cout << tensor_y.dim_size(i) << " ";
        }
        std::cout << std::endl;

        auto session = tensorflow::NewSession(tensorflow::SessionOptions());
        if (!session) {
            return 0;
        }

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto x_placeholder = tensorflow::ops::Placeholder(root.WithOpName("x"), tensorflow::DT_BOOL);
        auto y_placeholder = tensorflow::ops::Placeholder(root.WithOpName("y"), tensorflow::DT_BOOL);
        
        auto logical_or_op = tensorflow::ops::LogicalOr(root.WithOpName("logical_or"), x_placeholder, y_placeholder);

        tensorflow::GraphDef graph;
        tensorflow::Status status = root.ToGraphDef(&graph);
        if (!status.ok()) {
            session->Close();
            delete session;
            return 0;
        }

        status = session->Create(graph);
        if (!status.ok()) {
            session->Close();
            delete session;
            return 0;
        }

        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"x", tensor_x},
            {"y", tensor_y}
        };

        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"logical_or"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "LogicalOr operation completed successfully" << std::endl;
            std::cout << "Output shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
        }

        session->Close();
        delete session;

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}