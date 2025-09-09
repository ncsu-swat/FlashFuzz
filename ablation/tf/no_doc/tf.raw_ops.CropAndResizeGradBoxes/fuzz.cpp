#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
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
      dtype = tensorflow::DT_DOUBLE;
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
        
        if (size < 20) {
            return 0;
        }

        tensorflow::DataType grads_dtype = parseDataType(data[offset++]);
        uint8_t grads_rank = parseRank(data[offset++]);
        if (grads_rank != 4) {
            grads_rank = 4;
        }
        std::vector<int64_t> grads_shape = parseShape(data, offset, size, grads_rank);
        
        tensorflow::DataType image_dtype = parseDataType(data[offset++]);
        uint8_t image_rank = parseRank(data[offset++]);
        if (image_rank != 4) {
            image_rank = 4;
        }
        std::vector<int64_t> image_shape = parseShape(data, offset, size, image_rank);
        
        uint8_t boxes_rank = parseRank(data[offset++]);
        if (boxes_rank != 2) {
            boxes_rank = 2;
        }
        std::vector<int64_t> boxes_shape = parseShape(data, offset, size, boxes_rank);
        boxes_shape[1] = 4;
        
        uint8_t box_ind_rank = parseRank(data[offset++]);
        if (box_ind_rank != 1) {
            box_ind_rank = 1;
        }
        std::vector<int64_t> box_ind_shape = parseShape(data, offset, size, box_ind_rank);
        box_ind_shape[0] = boxes_shape[0];

        tensorflow::Tensor grads_tensor(grads_dtype, tensorflow::TensorShape(grads_shape));
        tensorflow::Tensor image_tensor(image_dtype, tensorflow::TensorShape(image_shape));
        tensorflow::Tensor boxes_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(boxes_shape));
        tensorflow::Tensor box_ind_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(box_ind_shape));

        fillTensorWithDataByType(grads_tensor, grads_dtype, data, offset, size);
        fillTensorWithDataByType(image_tensor, image_dtype, data, offset, size);
        fillTensorWithDataByType(boxes_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(box_ind_tensor, tensorflow::DT_INT32, data, offset, size);

        std::cout << "Grads tensor shape: ";
        for (int i = 0; i < grads_tensor.dims(); ++i) {
            std::cout << grads_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Image tensor shape: ";
        for (int i = 0; i < image_tensor.dims(); ++i) {
            std::cout << image_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Boxes tensor shape: ";
        for (int i = 0; i < boxes_tensor.dims(); ++i) {
            std::cout << boxes_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Box_ind tensor shape: ";
        for (int i = 0; i < box_ind_tensor.dims(); ++i) {
            std::cout << box_ind_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto grads_placeholder = tensorflow::ops::Placeholder(root, grads_dtype);
        auto image_placeholder = tensorflow::ops::Placeholder(root, image_dtype);
        auto boxes_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto box_ind_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);

        auto crop_and_resize_grad_boxes = tensorflow::ops::CropAndResizeGradBoxes(
            root, grads_placeholder, image_placeholder, boxes_placeholder, box_ind_placeholder);

        tensorflow::GraphDef graph;
        TF_CHECK_OK(root.ToGraphDef(&graph));

        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        TF_CHECK_OK(session->Create(graph));

        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {grads_placeholder.node()->name(), grads_tensor},
            {image_placeholder.node()->name(), image_tensor},
            {boxes_placeholder.node()->name(), boxes_tensor},
            {box_ind_placeholder.node()->name(), box_ind_tensor}
        };

        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session->Run(inputs, {crop_and_resize_grad_boxes.node()->name()}, {}, &outputs);

        if (status.ok() && !outputs.empty()) {
            std::cout << "CropAndResizeGradBoxes output shape: ";
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