#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

#define MAX_RANK 3
#define MIN_RANK 3
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 256

namespace tf_fuzzer_utils {
    void logError(const std::string& message, const uint8_t* data, size_t size) {
        std::cerr << "Error: " << message << std::endl;
    }
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
    case tensorflow::DT_UINT8:
      fillTensorWithData<uint8_t>(tensor, data, offset, total_size);
      break;
    default:
      break;
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

std::string parseFormat(uint8_t selector) {
    switch (selector % 3) {
        case 0: return "";
        case 1: return "grayscale";
        case 2: return "rgb";
        default: return "";
    }
}

std::string parseDensityUnit(uint8_t selector) {
    switch (selector % 2) {
        case 0: return "in";
        case 1: return "cm";
        default: return "in";
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 50) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t rank = parseRank(data[offset++]);
        std::vector<int64_t> shape = parseShape(data, offset, size, rank);
        
        tensorflow::TensorShape tensor_shape;
        for (int64_t dim : shape) {
            tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor image_tensor(tensorflow::DT_UINT8, tensor_shape);
        fillTensorWithDataByType(image_tensor, tensorflow::DT_UINT8, data, offset, size);
        
        if (offset >= size) return 0;
        
        std::string format = parseFormat(data[offset++]);
        
        if (offset >= size) return 0;
        int quality = 95;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&quality, data + offset, sizeof(int));
            offset += sizeof(int);
            quality = std::abs(quality) % 101;
        }
        
        if (offset >= size) return 0;
        bool progressive = (data[offset++] % 2) == 1;
        
        if (offset >= size) return 0;
        bool optimize_size = (data[offset++] % 2) == 1;
        
        if (offset >= size) return 0;
        bool chroma_downsampling = (data[offset++] % 2) == 1;
        
        if (offset >= size) return 0;
        std::string density_unit = parseDensityUnit(data[offset++]);
        
        if (offset >= size) return 0;
        int x_density = 300;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&x_density, data + offset, sizeof(int));
            offset += sizeof(int);
            x_density = std::abs(x_density) % 1000 + 1;
        }
        
        if (offset >= size) return 0;
        int y_density = 300;
        if (offset + sizeof(int) <= size) {
            std::memcpy(&y_density, data + offset, sizeof(int));
            offset += sizeof(int);
            y_density = std::abs(y_density) % 1000 + 1;
        }
        
        std::string xmp_metadata = "";
        if (offset < size) {
            size_t remaining = size - offset;
            if (remaining > 0 && remaining < 1000) {
                xmp_metadata = std::string(reinterpret_cast<const char*>(data + offset), remaining);
            }
        }

        auto image_input = tensorflow::ops::Placeholder(root, tensorflow::DT_UINT8);
        
        auto encode_jpeg = tensorflow::ops::EncodeJpeg(
            root, 
            image_input,
            tensorflow::ops::EncodeJpeg::Format(format)
                .Quality(quality)
                .Progressive(progressive)
                .OptimizeSize(optimize_size)
                .ChromaDownsampling(chroma_downsampling)
                .DensityUnit(density_unit)
                .XDensity(x_density)
                .YDensity(y_density)
                .XmpMetadata(xmp_metadata)
        );

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({{image_input, image_tensor}}, {encode_jpeg}, &outputs);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
