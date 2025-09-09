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
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 6) {
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
            dtype = tensorflow::DT_INT32;
            break;
        case 3:
            dtype = tensorflow::DT_INT64;
            break;
        case 4:
            dtype = tensorflow::DT_HALF;
            break;
        case 5:
            dtype = tensorflow::DT_BFLOAT16;
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
      return;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType values_dtype = parseDataType(data[offset++]);
        uint8_t values_rank = parseRank(data[offset++]);
        std::vector<int64_t> values_shape = parseShape(data, offset, size, values_rank);
        
        tensorflow::DataType range_dtype = parseDataType(data[offset++]);
        uint8_t range_rank = parseRank(data[offset++]);
        std::vector<int64_t> range_shape = parseShape(data, offset, size, range_rank);
        
        int32_t nbins = 5;
        if (offset + sizeof(int32_t) <= size) {
            std::memcpy(&nbins, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            nbins = std::abs(nbins) % 100 + 1;
        }

        tensorflow::TensorShape values_tensor_shape;
        for (int64_t dim : values_shape) {
            values_tensor_shape.AddDim(dim);
        }
        
        tensorflow::TensorShape range_tensor_shape;
        for (int64_t dim : range_shape) {
            range_tensor_shape.AddDim(dim);
        }

        tensorflow::Tensor values_tensor(values_dtype, values_tensor_shape);
        tensorflow::Tensor range_tensor(range_dtype, range_tensor_shape);
        tensorflow::Tensor nbins_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        
        fillTensorWithDataByType(values_tensor, values_dtype, data, offset, size);
        fillTensorWithDataByType(range_tensor, range_dtype, data, offset, size);
        
        nbins_tensor.scalar<int32_t>()() = nbins;

        std::cout << "Values tensor shape: ";
        for (int i = 0; i < values_tensor.dims(); ++i) {
            std::cout << values_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Range tensor shape: ";
        for (int i = 0; i < range_tensor.dims(); ++i) {
            std::cout << range_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Nbins: " << nbins << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto values_placeholder = tensorflow::ops::Placeholder(root.WithOpName("values"), values_dtype);
        auto range_placeholder = tensorflow::ops::Placeholder(root.WithOpName("range"), range_dtype);
        auto nbins_placeholder = tensorflow::ops::Placeholder(root.WithOpName("nbins"), tensorflow::DT_INT32);
        
        auto histogram_op = tensorflow::ops::HistogramFixedWidth(root.WithOpName("histogram"), 
                                                               values_placeholder, 
                                                               range_placeholder, 
                                                               nbins_placeholder);

        tensorflow::GraphDef graph;
        tensorflow::Status status = root.ToGraphDef(&graph);
        if (!status.ok()) {
            std::cout << "Failed to create graph: " << status.ToString() << std::endl;
            return 0;
        }

        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = session->Create(graph);
        if (!status.ok()) {
            std::cout << "Failed to create session: " << status.ToString() << std::endl;
            return 0;
        }

        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"values", values_tensor},
            {"range", range_tensor},
            {"nbins", nbins_tensor}
        };

        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"histogram"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "HistogramFixedWidth operation completed successfully" << std::endl;
            std::cout << "Output shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "HistogramFixedWidth operation failed: " << status.ToString() << std::endl;
        }

        session->Close();

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}