#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/audio_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
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

        uint8_t spectrogram_rank = parseRank(data[offset++]);
        std::vector<int64_t> spectrogram_shape = parseShape(data, offset, size, spectrogram_rank);
        
        uint8_t sample_rate_rank = parseRank(data[offset++]);
        std::vector<int64_t> sample_rate_shape = parseShape(data, offset, size, sample_rate_rank);

        if (offset >= size) {
            return 0;
        }

        tensorflow::Tensor spectrogram_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(spectrogram_shape));
        fillTensorWithDataByType(spectrogram_tensor, tensorflow::DT_FLOAT, data, offset, size);

        tensorflow::Tensor sample_rate_tensor(tensorflow::DT_INT32, tensorflow::TensorShape(sample_rate_shape));
        fillTensorWithDataByType(sample_rate_tensor, tensorflow::DT_INT32, data, offset, size);

        float upper_frequency_limit = 4000.0f;
        float lower_frequency_limit = 20.0f;
        int filterbank_channel_count = 40;
        int dct_coefficient_count = 13;

        if (offset + sizeof(float) <= size) {
            std::memcpy(&upper_frequency_limit, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        if (offset + sizeof(float) <= size) {
            std::memcpy(&lower_frequency_limit, data + offset, sizeof(float));
            offset += sizeof(float);
        }
        if (offset + sizeof(int) <= size) {
            std::memcpy(&filterbank_channel_count, data + offset, sizeof(int));
            offset += sizeof(int);
            filterbank_channel_count = std::abs(filterbank_channel_count % 100) + 1;
        }
        if (offset + sizeof(int) <= size) {
            std::memcpy(&dct_coefficient_count, data + offset, sizeof(int));
            offset += sizeof(int);
            dct_coefficient_count = std::abs(dct_coefficient_count % 100) + 1;
        }

        std::cout << "Spectrogram tensor shape: ";
        for (int i = 0; i < spectrogram_tensor.dims(); ++i) {
            std::cout << spectrogram_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Sample rate tensor shape: ";
        for (int i = 0; i < sample_rate_tensor.dims(); ++i) {
            std::cout << sample_rate_tensor.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Upper frequency limit: " << upper_frequency_limit << std::endl;
        std::cout << "Lower frequency limit: " << lower_frequency_limit << std::endl;
        std::cout << "Filterbank channel count: " << filterbank_channel_count << std::endl;
        std::cout << "DCT coefficient count: " << dct_coefficient_count << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto spectrogram_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto sample_rate_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);

        auto mfcc_op = tensorflow::ops::Mfcc(root, spectrogram_placeholder, sample_rate_placeholder,
                                           tensorflow::ops::Mfcc::Attrs()
                                               .UpperFrequencyLimit(upper_frequency_limit)
                                               .LowerFrequencyLimit(lower_frequency_limit)
                                               .FilterbankChannelCount(filterbank_channel_count)
                                               .DctCoefficientCount(dct_coefficient_count));

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({{spectrogram_placeholder, spectrogram_tensor},
                                                 {sample_rate_placeholder, sample_rate_tensor}},
                                               {mfcc_op}, &outputs);

        if (status.ok() && !outputs.empty()) {
            std::cout << "MFCC operation successful. Output shape: ";
            for (int i = 0; i < outputs[0].dims(); ++i) {
                std::cout << outputs[0].dim_size(i) << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "MFCC operation failed: " << status.ToString() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}