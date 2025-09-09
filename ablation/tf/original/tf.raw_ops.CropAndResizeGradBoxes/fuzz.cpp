#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/core/public/session.h>

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

tensorflow::DataType parseImageDataType(uint8_t selector) {
  tensorflow::DataType dtype; 
  switch (selector % 9) {  
    case 0:
      dtype = tensorflow::DT_UINT8;
      break;
    case 1:
      dtype = tensorflow::DT_UINT16;
      break;
    case 2:
      dtype = tensorflow::DT_INT8;
      break;
    case 3:
      dtype = tensorflow::DT_INT16;
      break;
    case 4:
      dtype = tensorflow::DT_INT32;
      break;
    case 5:
      dtype = tensorflow::DT_INT64;
      break;
    case 6:
      dtype = tensorflow::DT_HALF;
      break;
    case 7:
      dtype = tensorflow::DT_FLOAT;
      break;
    case 8:
      dtype = tensorflow::DT_DOUBLE;
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

        uint8_t image_dtype_selector = data[offset++];
        tensorflow::DataType image_dtype = parseImageDataType(image_dtype_selector);
        
        std::vector<int64_t> grads_shape = {2, 3, 3, 2};
        tensorflow::TensorShape grads_tensor_shape(grads_shape);
        tensorflow::Tensor grads_tensor(tensorflow::DT_FLOAT, grads_tensor_shape);
        fillTensorWithDataByType(grads_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        std::vector<int64_t> image_shape = {1, 5, 5, 2};
        tensorflow::TensorShape image_tensor_shape(image_shape);
        tensorflow::Tensor image_tensor(image_dtype, image_tensor_shape);
        fillTensorWithDataByType(image_tensor, image_dtype, data, offset, size);
        
        std::vector<int64_t> boxes_shape = {2, 4};
        tensorflow::TensorShape boxes_tensor_shape(boxes_shape);
        tensorflow::Tensor boxes_tensor(tensorflow::DT_FLOAT, boxes_tensor_shape);
        fillTensorWithDataByType(boxes_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        std::vector<int64_t> box_ind_shape = {2};
        tensorflow::TensorShape box_ind_tensor_shape(box_ind_shape);
        tensorflow::Tensor box_ind_tensor(tensorflow::DT_INT32, box_ind_tensor_shape);
        fillTensorWithDataByType(box_ind_tensor, tensorflow::DT_INT32, data, offset, size);
        
        std::cout << "grads shape: ";
        for (int i = 0; i < grads_tensor.dims(); ++i) {
            std::cout << grads_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "image shape: ";
        for (int i = 0; i < image_tensor.dims(); ++i) {
            std::cout << image_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "boxes shape: ";
        for (int i = 0; i < boxes_tensor.dims(); ++i) {
            std::cout << boxes_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "box_ind shape: ";
        for (int i = 0; i < box_ind_tensor.dims(); ++i) {
            std::cout << box_ind_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto grads_op = tensorflow::ops::Const(root, grads_tensor);
        auto image_op = tensorflow::ops::Const(root, image_tensor);
        auto boxes_op = tensorflow::ops::Const(root, boxes_tensor);
        auto box_ind_op = tensorflow::ops::Const(root, box_ind_tensor);
        
        auto crop_and_resize_grad_boxes = tensorflow::ops::CropAndResizeGradBoxes(
            root, grads_op, image_op, boxes_op, box_ind_op,
            tensorflow::ops::CropAndResizeGradBoxes::Method("bilinear"));
        
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({crop_and_resize_grad_boxes}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            std::cout << "Operation executed successfully" << std::endl;
            std::cout << "Output shape: ";
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