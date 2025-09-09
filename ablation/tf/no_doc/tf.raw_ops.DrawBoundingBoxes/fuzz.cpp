#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

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

tensorflow::DataType parseDataType(uint8_t selector) {
  tensorflow::DataType dtype; 
  switch (selector % 3) {  
    case 0:
      dtype = tensorflow::DT_FLOAT;
      break;
    case 1:
      dtype = tensorflow::DT_UINT8;
      break;
    case 2:
      dtype = tensorflow::DT_HALF;
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

        tensorflow::DataType images_dtype = parseDataType(data[offset++]);
        uint8_t images_rank = parseRank(data[offset++]);
        if (images_rank < 3) images_rank = 4;
        std::vector<int64_t> images_shape = parseShape(data, offset, size, images_rank);
        
        if (images_shape.size() < 4) {
            images_shape = {1, 10, 10, 3};
        }
        
        tensorflow::TensorShape images_tensor_shape;
        for (auto dim : images_shape) {
            images_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor images_tensor(images_dtype, images_tensor_shape);
        fillTensorWithDataByType(images_tensor, images_dtype, data, offset, size);
        
        uint8_t boxes_rank = parseRank(data[offset++]);
        if (boxes_rank < 3) boxes_rank = 3;
        std::vector<int64_t> boxes_shape = parseShape(data, offset, size, boxes_rank);
        
        if (boxes_shape.size() < 3) {
            boxes_shape = {1, 1, 4};
        } else {
            boxes_shape[boxes_shape.size()-1] = 4;
        }
        
        tensorflow::TensorShape boxes_tensor_shape;
        for (auto dim : boxes_shape) {
            boxes_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor boxes_tensor(tensorflow::DT_FLOAT, boxes_tensor_shape);
        fillTensorWithDataByType(boxes_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        auto boxes_flat = boxes_tensor.flat<float>();
        for (int i = 0; i < boxes_flat.size(); ++i) {
            if (boxes_flat(i) < 0.0f) boxes_flat(i) = 0.0f;
            if (boxes_flat(i) > 1.0f) boxes_flat(i) = 1.0f;
        }
        
        std::cout << "Images tensor shape: ";
        for (int i = 0; i < images_tensor.dims(); ++i) {
            std::cout << images_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Boxes tensor shape: ";
        for (int i = 0; i < boxes_tensor.dims(); ++i) {
            std::cout << boxes_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto images_const = tensorflow::ops::Const(root, images_tensor);
        auto boxes_const = tensorflow::ops::Const(root, boxes_tensor);
        
        auto draw_bounding_boxes = tensorflow::ops::DrawBoundingBoxes(root, images_const, boxes_const);
        
        tensorflow::GraphDef graph;
        TF_CHECK_OK(root.ToGraphDef(&graph));
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        TF_CHECK_OK(session->Create(graph));
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status run_status = session->Run({}, {draw_bounding_boxes.node()->name() + ":0"}, {}, &outputs);
        
        if (run_status.ok() && !outputs.empty()) {
            std::cout << "DrawBoundingBoxes executed successfully" << std::endl;
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "DrawBoundingBoxes failed: " << run_status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}