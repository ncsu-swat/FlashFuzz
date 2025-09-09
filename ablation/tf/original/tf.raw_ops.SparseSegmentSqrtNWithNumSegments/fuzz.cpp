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
#include <tensorflow/core/kernels/ops_util.h>

constexpr uint8_t MIN_RANK = 1;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

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
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType data_dtype = parseDataType(data[offset++]);
        tensorflow::DataType indices_dtype = parseIndicesDataType(data[offset++]);
        tensorflow::DataType segment_ids_dtype = parseIndicesDataType(data[offset++]);
        tensorflow::DataType num_segments_dtype = parseIndicesDataType(data[offset++]);
        
        uint8_t data_rank = parseRank(data[offset++]);
        
        std::vector<int64_t> data_shape = parseShape(data, offset, size, data_rank);
        
        if (offset >= size) {
            return 0;
        }
        
        int64_t indices_size = 1 + (data[offset++] % 10);
        int64_t num_segments_val = 1 + (data[offset++] % 5);
        
        bool sparse_gradient = (data[offset++] % 2) == 1;
        
        tensorflow::Tensor data_tensor(data_dtype, tensorflow::TensorShape(data_shape));
        fillTensorWithDataByType(data_tensor, data_dtype, data, offset, size);
        
        tensorflow::Tensor indices_tensor(indices_dtype, tensorflow::TensorShape({indices_size}));
        if (indices_dtype == tensorflow::DT_INT32) {
            auto indices_flat = indices_tensor.flat<int32_t>();
            for (int64_t i = 0; i < indices_size; ++i) {
                if (offset < size) {
                    int32_t val = static_cast<int32_t>(data[offset++] % data_shape[0]);
                    indices_flat(i) = val;
                } else {
                    indices_flat(i) = 0;
                }
            }
        } else {
            auto indices_flat = indices_tensor.flat<int64_t>();
            for (int64_t i = 0; i < indices_size; ++i) {
                if (offset < size) {
                    int64_t val = static_cast<int64_t>(data[offset++] % data_shape[0]);
                    indices_flat(i) = val;
                } else {
                    indices_flat(i) = 0;
                }
            }
        }
        
        tensorflow::Tensor segment_ids_tensor(segment_ids_dtype, tensorflow::TensorShape({indices_size}));
        if (segment_ids_dtype == tensorflow::DT_INT32) {
            auto segment_ids_flat = segment_ids_tensor.flat<int32_t>();
            for (int64_t i = 0; i < indices_size; ++i) {
                if (offset < size) {
                    int32_t val = static_cast<int32_t>(data[offset++] % num_segments_val);
                    segment_ids_flat(i) = val;
                } else {
                    segment_ids_flat(i) = 0;
                }
            }
        } else {
            auto segment_ids_flat = segment_ids_tensor.flat<int64_t>();
            for (int64_t i = 0; i < indices_size; ++i) {
                if (offset < size) {
                    int64_t val = static_cast<int64_t>(data[offset++] % num_segments_val);
                    segment_ids_flat(i) = val;
                } else {
                    segment_ids_flat(i) = 0;
                }
            }
        }
        
        tensorflow::Tensor num_segments_tensor(num_segments_dtype, tensorflow::TensorShape({}));
        if (num_segments_dtype == tensorflow::DT_INT32) {
            num_segments_tensor.scalar<int32_t>()() = static_cast<int32_t>(num_segments_val);
        } else {
            num_segments_tensor.scalar<int64_t>()() = num_segments_val;
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
        
        std::cout << "Num segments: " << num_segments_val << std::endl;
        std::cout << "Sparse gradient: " << sparse_gradient << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto data_op = tensorflow::ops::Const(root, data_tensor);
        auto indices_op = tensorflow::ops::Const(root, indices_tensor);
        auto segment_ids_op = tensorflow::ops::Const(root, segment_ids_tensor);
        auto num_segments_op = tensorflow::ops::Const(root, num_segments_tensor);
        
        tensorflow::Node* sparse_segment_sqrt_n_node;
        tensorflow::NodeBuilder node_builder("SparseSegmentSqrtNWithNumSegments", "SparseSegmentSqrtNWithNumSegments");
        node_builder.Input(data_op.node())
                   .Input(indices_op.node())
                   .Input(segment_ids_op.node())
                   .Input(num_segments_op.node())
                   .Attr("sparse_gradient", sparse_gradient);
        
        tensorflow::Status status = node_builder.Finalize(root.graph(), &sparse_segment_sqrt_n_node);
        if (!status.ok()) {
            std::cout << "Node creation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({tensorflow::Output(sparse_segment_sqrt_n_node)}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "Operation executed successfully" << std::endl;
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}