#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/audio_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10

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
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        uint8_t spectrogram_rank = parseRank(data[offset++]);
        std::vector<int64_t> spectrogram_shape = parseShape(data, offset, size, spectrogram_rank);
        
        tensorflow::TensorShape spectrogram_tensor_shape;
        for (int64_t dim : spectrogram_shape) {
            spectrogram_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor spectrogram_tensor(tensorflow::DT_FLOAT, spectrogram_tensor_shape);
        fillTensorWithDataByType(spectrogram_tensor, tensorflow::DT_FLOAT, data, offset, size);
        
        uint8_t sample_rate_rank = parseRank(data[offset++]);
        std::vector<int64_t> sample_rate_shape = parseShape(data, offset, size, sample_rate_rank);
        
        tensorflow::TensorShape sample_rate_tensor_shape;
        for (int64_t dim : sample_rate_shape) {
            sample_rate_tensor_shape.AddDim(dim);
        }
        
        tensorflow::Tensor sample_rate_tensor(tensorflow::DT_INT32, sample_rate_tensor_shape);
        fillTensorWithDataByType(sample_rate_tensor, tensorflow::DT_INT32, data, offset, size);
        
        float upper_frequency_limit = 4000.0f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&upper_frequency_limit, data + offset, sizeof(float));
            offset += sizeof(float);
            if (upper_frequency_limit < 0) upper_frequency_limit = 4000.0f;
        }
        
        float lower_frequency_limit = 20.0f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&lower_frequency_limit, data + offset, sizeof(float));
            offset += sizeof(float);
            if (lower_frequency_limit < 0) lower_frequency_limit = 20.0f;
        }
        
        int64_t filterbank_channel_count = 40;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&filterbank_channel_count, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            filterbank_channel_count = std::abs(filterbank_channel_count) % 100 + 1;
        }
        
        int64_t dct_coefficient_count = 13;
        if (offset + sizeof(int64_t) <= size) {
            std::memcpy(&dct_coefficient_count, data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dct_coefficient_count = std::abs(dct_coefficient_count) % 100 + 1;
        }

        auto spectrogram_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto sample_rate_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);

        auto mfcc_op = tensorflow::ops::Mfcc(
            root,
            spectrogram_placeholder,
            sample_rate_placeholder,
            tensorflow::ops::Mfcc::UpperFrequencyLimit(upper_frequency_limit)
                .LowerFrequencyLimit(lower_frequency_limit)
                .FilterbankChannelCount(filterbank_channel_count)
                .DctCoefficientCount(dct_coefficient_count)
        );

        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{spectrogram_placeholder, spectrogram_tensor}, {sample_rate_placeholder, sample_rate_tensor}},
            {mfcc_op},
            &outputs
        );
        
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
