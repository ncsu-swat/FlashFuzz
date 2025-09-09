#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

constexpr uint8_t MIN_RANK = 1;
constexpr uint8_t MAX_RANK = 6;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 100;

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
  switch (selector % 15) {  
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
      dtype = tensorflow::DT_UINT8;
      break;
    case 4:
      dtype = tensorflow::DT_INT16;
      break;
    case 5:
      dtype = tensorflow::DT_INT8;
      break;
    case 6:
      dtype = tensorflow::DT_COMPLEX64;
      break;
    case 7:
      dtype = tensorflow::DT_INT64;
      break;
    case 8:
      dtype = tensorflow::DT_BOOL;
      break;
    case 9:
      dtype = tensorflow::DT_BFLOAT16;
      break;
    case 10:
      dtype = tensorflow::DT_UINT16;
      break;
    case 11:
      dtype = tensorflow::DT_COMPLEX128;
      break;
    case 12:
      dtype = tensorflow::DT_HALF;
      break;
    case 13:
      dtype = tensorflow::DT_UINT32;
      break;
    case 14:
      dtype = tensorflow::DT_UINT64;
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

        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        uint8_t input_rank = parseRank(data[offset++]);
        
        if (input_rank < 2) {
            input_rank = 2;
        }
        
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        if (input_shape.empty()) {
            return 0;
        }
        
        int64_t spatial_dims = input_rank - 2;
        if (spatial_dims <= 0) {
            return 0;
        }
        
        std::vector<int64_t> block_shape_dims = {spatial_dims};
        std::vector<int64_t> crops_dims = {spatial_dims, 2};
        
        tensorflow::TensorShape input_tensor_shape(input_shape);
        tensorflow::TensorShape block_shape_tensor_shape(block_shape_dims);
        tensorflow::TensorShape crops_tensor_shape(crops_dims);
        
        tensorflow::Tensor input_tensor(input_dtype, input_tensor_shape);
        tensorflow::Tensor block_shape_tensor(tensorflow::DT_INT32, block_shape_tensor_shape);
        tensorflow::Tensor crops_tensor(tensorflow::DT_INT32, crops_tensor_shape);
        
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);
        
        auto block_shape_flat = block_shape_tensor.flat<int32_t>();
        for (int64_t i = 0; i < spatial_dims; ++i) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t block_val;
                std::memcpy(&block_val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                block_val = std::abs(block_val) % 10 + 1;
                block_shape_flat(i) = block_val;
            } else {
                block_shape_flat(i) = 1;
            }
        }
        
        auto crops_flat = crops_tensor.flat<int32_t>();
        for (int64_t i = 0; i < spatial_dims * 2; ++i) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t crop_val;
                std::memcpy(&crop_val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                crop_val = std::abs(crop_val) % 5;
                crops_flat(i) = crop_val;
            } else {
                crops_flat(i) = 0;
            }
        }
        
        std::cout << "Input tensor shape: ";
        for (int i = 0; i < input_tensor.dims(); ++i) {
            std::cout << input_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Block shape: ";
        for (int64_t i = 0; i < spatial_dims; ++i) {
            std::cout << block_shape_flat(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Crops: ";
        for (int64_t i = 0; i < spatial_dims * 2; ++i) {
            std::cout << crops_flat(i) << " ";
        }
        std::cout << std::endl;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, input_dtype);
        auto block_shape_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto crops_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        
        auto batch_to_space_nd = tensorflow::ops::BatchToSpaceND(root, input_placeholder, block_shape_placeholder, crops_placeholder);
        
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({{input_placeholder, input_tensor}, 
                                                 {block_shape_placeholder, block_shape_tensor}, 
                                                 {crops_placeholder, crops_tensor}}, 
                                                {batch_to_space_nd}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
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