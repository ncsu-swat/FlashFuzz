#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/public/version.h>
#include <tensorflow/cc/ops/math_ops.h>

constexpr uint8_t MIN_RANK = 1;
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

tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype; 
  switch (selector % 8) {  
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
      dtype = tensorflow::DT_BFLOAT16;
      break;
    case 5:
      dtype = tensorflow::DT_HALF;
      break;
    case 6:
      dtype = tensorflow::DT_COMPLEX64;
      break;
    case 7:
      dtype = tensorflow::DT_COMPLEX128;
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType data_dtype = parseDataType(data[offset++]);
        
        uint8_t data_rank = parseRank(data[offset++]);
        std::vector<int64_t> data_shape = parseShape(data, offset, size, data_rank);
        
        uint8_t indices_rank = 1;
        std::vector<int64_t> indices_shape = {std::min(static_cast<int64_t>(5), data_shape[0])};
        
        uint8_t segment_ids_rank = 1;
        std::vector<int64_t> segment_ids_shape = indices_shape;
        
        uint8_t num_segments_rank = 0;
        std::vector<int64_t> num_segments_shape = {};

        tensorflow::TensorShape data_tensor_shape(data_shape);
        tensorflow::TensorShape indices_tensor_shape(indices_shape);
        tensorflow::TensorShape segment_ids_tensor_shape(segment_ids_shape);
        tensorflow::TensorShape num_segments_tensor_shape(num_segments_shape);

        tensorflow::Tensor data_tensor(data_dtype, data_tensor_shape);
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, indices_tensor_shape);
        tensorflow::Tensor segment_ids_tensor(tensorflow::DT_INT32, segment_ids_tensor_shape);
        tensorflow::Tensor num_segments_tensor(tensorflow::DT_INT32, num_segments_tensor_shape);

        fillTensorWithDataByType(data_tensor, data_dtype, data, offset, size);
        
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < indices_flat.size(); ++i) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t val;
                std::memcpy(&val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                indices_flat(i) = std::abs(val) % static_cast<int32_t>(data_shape[0]);
            } else {
                indices_flat(i) = i % static_cast<int32_t>(data_shape[0]);
            }
        }

        auto segment_ids_flat = segment_ids_tensor.flat<int32_t>();
        for (int i = 0; i < segment_ids_flat.size(); ++i) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t val;
                std::memcpy(&val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                segment_ids_flat(i) = std::abs(val) % 3;
            } else {
                segment_ids_flat(i) = i % 3;
            }
        }

        auto num_segments_scalar = num_segments_tensor.scalar<int32_t>();
        if (offset + sizeof(int32_t) <= size) {
            int32_t val;
            std::memcpy(&val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            num_segments_scalar() = std::max(1, std::abs(val) % 10);
        } else {
            num_segments_scalar() = 3;
        }

        std::cout << "Data tensor shape: ";
        for (int i = 0; i < data_tensor.dims(); ++i) {
            std::cout << data_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Indices tensor shape: ";
        for (int i = 0; i < indices_tensor.dims(); ++i) {
            std::cout << indices_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Segment IDs tensor shape: ";
        for (int i = 0; i < segment_ids_tensor.dims(); ++i) {
            std::cout << segment_ids_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Num segments: " << num_segments_scalar() << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto data_placeholder = tensorflow::ops::Placeholder(root, data_dtype);
        auto indices_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto segment_ids_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto num_segments_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);

        tensorflow::Node* sparse_segment_mean_node;
        tensorflow::NodeBuilder node_builder("SparseSegmentMeanWithNumSegments", "SparseSegmentMeanWithNumSegments");
        node_builder.Input(data_placeholder.node())
                   .Input(indices_placeholder.node())
                   .Input(segment_ids_placeholder.node())
                   .Input(num_segments_placeholder.node())
                   .Attr("T", data_dtype);
        
        tensorflow::Status status = node_builder.Finalize(root.graph(), &sparse_segment_mean_node);
        if (!status.ok()) {
            std::cout << "Node creation failed: " << status.ToString() << std::endl;
            return 0;
        }

        tensorflow::GraphDef graph_def;
        status = root.ToGraphDef(&graph_def);
        if (!status.ok()) {
            std::cout << "Graph creation failed: " << status.ToString() << std::endl;
            return 0;
        }

        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = session->Create(graph_def);
        if (!status.ok()) {
            std::cout << "Session creation failed: " << status.ToString() << std::endl;
            return 0;
        }

        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {data_placeholder.node()->name(), data_tensor},
            {indices_placeholder.node()->name(), indices_tensor},
            {segment_ids_placeholder.node()->name(), segment_ids_tensor},
            {num_segments_placeholder.node()->name(), num_segments_tensor}
        };

        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {sparse_segment_mean_node->name()};

        status = session->Run(inputs, output_names, {}, &outputs);
        if (!status.ok()) {
            std::cout << "Session run failed: " << status.ToString() << std::endl;
            return 0;
        }

        if (!outputs.empty()) {
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
        }

        session->Close();

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}