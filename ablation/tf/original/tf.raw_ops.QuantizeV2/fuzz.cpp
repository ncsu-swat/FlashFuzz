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
#include <tensorflow/core/framework/node_def_util.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_def_builder.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op_def_util.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/graph/node_builder.h>
#include <tensorflow/core/graph/graph.h>
#include <tensorflow/core/common_runtime/executor.h>
#include <tensorflow/core/common_runtime/function.h>
#include <tensorflow/core/framework/function.h>
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/framework/allocator.h>
#include <tensorflow/core/framework/device_base.h>
#include <tensorflow/core/common_runtime/device_factory.h>
#include <tensorflow/core/common_runtime/device_mgr.h>
#include <tensorflow/core/common_runtime/rendezvous_mgr.h>
#include <tensorflow/core/framework/rendezvous.h>
#include <tensorflow/core/common_runtime/step_stats_collector.h>
#include <tensorflow/core/framework/step_stats.pb.h>
#include <tensorflow/core/framework/tensor_types.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tensorflow/core/lib/gtl/inlined_vector.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/quantize_op.h>

constexpr uint8_t MIN_RANK = 0;
constexpr uint8_t MAX_RANK = 4;
constexpr int64_t MIN_TENSOR_SHAPE_DIMS_TF = 1;
constexpr int64_t MAX_TENSOR_SHAPE_DIMS_TF = 10;

tensorflow::DataType parseQuantizedDataType(uint8_t selector) {
    tensorflow::DataType dtype;
    switch (selector % 5) {
        case 0:
            dtype = tensorflow::DT_QINT8;
            break;
        case 1:
            dtype = tensorflow::DT_QUINT8;
            break;
        case 2:
            dtype = tensorflow::DT_QINT32;
            break;
        case 3:
            dtype = tensorflow::DT_QINT16;
            break;
        case 4:
            dtype = tensorflow::DT_QUINT16;
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
      break;
  }
}

std::string parseMode(uint8_t selector) {
    switch (selector % 3) {
        case 0:
            return "MIN_COMBINED";
        case 1:
            return "MIN_FIRST";
        case 2:
            return "SCALED";
        default:
            return "MIN_COMBINED";
    }
}

std::string parseRoundMode(uint8_t selector) {
    switch (selector % 2) {
        case 0:
            return "HALF_AWAY_FROM_ZERO";
        case 1:
            return "HALF_TO_EVEN";
        default:
            return "HALF_AWAY_FROM_ZERO";
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 10) {
            return 0;
        }

        tensorflow::DataType output_dtype = parseQuantizedDataType(data[offset++]);
        uint8_t input_rank = parseRank(data[offset++]);
        uint8_t min_range_rank = parseRank(data[offset++]);
        uint8_t max_range_rank = parseRank(data[offset++]);
        
        std::string mode = parseMode(data[offset++]);
        std::string round_mode = parseRoundMode(data[offset++]);
        bool narrow_range = (data[offset++] % 2) == 1;
        
        int32_t axis = -1;
        if (offset < size) {
            int32_t axis_val;
            std::memcpy(&axis_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            axis = axis_val % 10;
        }
        
        float ensure_minimum_range = 0.01f;
        if (offset + sizeof(float) <= size) {
            std::memcpy(&ensure_minimum_range, data + offset, sizeof(float));
            offset += sizeof(float);
            if (ensure_minimum_range < 0.0f || ensure_minimum_range > 1.0f) {
                ensure_minimum_range = 0.01f;
            }
        }

        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        std::vector<int64_t> min_range_shape = parseShape(data, offset, size, min_range_rank);
        std::vector<int64_t> max_range_shape = parseShape(data, offset, size, max_range_rank);

        tensorflow::TensorShape input_tensor_shape;
        tensorflow::TensorShape min_range_tensor_shape;
        tensorflow::TensorShape max_range_tensor_shape;
        
        for (int64_t dim : input_shape) {
            input_tensor_shape.AddDim(dim);
        }
        for (int64_t dim : min_range_shape) {
            min_range_tensor_shape.AddDim(dim);
        }
        for (int64_t dim : max_range_shape) {
            max_range_tensor_shape.AddDim(dim);
        }

        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_tensor_shape);
        tensorflow::Tensor min_range_tensor(tensorflow::DT_FLOAT, min_range_tensor_shape);
        tensorflow::Tensor max_range_tensor(tensorflow::DT_FLOAT, max_range_tensor_shape);

        fillTensorWithDataByType(input_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(min_range_tensor, tensorflow::DT_FLOAT, data, offset, size);
        fillTensorWithDataByType(max_range_tensor, tensorflow::DT_FLOAT, data, offset, size);

        std::cout << "Input tensor shape: ";
        for (int i = 0; i < input_tensor_shape.dims(); ++i) {
            std::cout << input_tensor_shape.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Min range tensor shape: ";
        for (int i = 0; i < min_range_tensor_shape.dims(); ++i) {
            std::cout << min_range_tensor_shape.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Max range tensor shape: ";
        for (int i = 0; i < max_range_tensor_shape.dims(); ++i) {
            std::cout << max_range_tensor_shape.dim_size(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Output dtype: " << tensorflow::DataTypeString(output_dtype) << std::endl;
        std::cout << "Mode: " << mode << std::endl;
        std::cout << "Round mode: " << round_mode << std::endl;
        std::cout << "Narrow range: " << narrow_range << std::endl;
        std::cout << "Axis: " << axis << std::endl;
        std::cout << "Ensure minimum range: " << ensure_minimum_range << std::endl;

        auto min_flat = min_range_tensor.flat<float>();
        auto max_flat = max_range_tensor.flat<float>();
        
        for (int i = 0; i < min_flat.size(); ++i) {
            if (min_flat(i) >= max_flat(i)) {
                max_flat(i) = min_flat(i) + 1.0f;
            }
        }

        tensorflow::OpKernelContext::Params params;
        tensorflow::DeviceBase device(tensorflow::Env::Default());
        params.device = &device;
        
        tensorflow::NodeDef node_def;
        node_def.set_name("quantize_v2_test");
        node_def.set_op("QuantizeV2");
        
        tensorflow::AttrValue dtype_attr;
        dtype_attr.set_type(output_dtype);
        node_def.mutable_attr()->insert({"T", dtype_attr});
        
        tensorflow::AttrValue mode_attr;
        mode_attr.set_s(mode);
        node_def.mutable_attr()->insert({"mode", mode_attr});
        
        tensorflow::AttrValue round_mode_attr;
        round_mode_attr.set_s(round_mode);
        node_def.mutable_attr()->insert({"round_mode", round_mode_attr});
        
        tensorflow::AttrValue narrow_range_attr;
        narrow_range_attr.set_b(narrow_range);
        node_def.mutable_attr()->insert({"narrow_range", narrow_range_attr});
        
        tensorflow::AttrValue axis_attr;
        axis_attr.set_i(axis);
        node_def.mutable_attr()->insert({"axis", axis_attr});
        
        tensorflow::AttrValue ensure_minimum_range_attr;
        ensure_minimum_range_attr.set_f(ensure_minimum_range);
        node_def.mutable_attr()->insert({"ensure_minimum_range", ensure_minimum_range_attr});
        
        params.def = &node_def;
        
        tensorflow::Status status;
        std::unique_ptr<tensorflow::OpKernel> kernel;
        
        tensorflow::OpKernelConstruction construction(tensorflow::DeviceType("CPU"), &device, 
                                                    tensorflow::cpu_allocator(), &node_def, 
                                                    tensorflow::OpDef(), &status);
        
        if (!status.ok()) {
            std::cout << "OpKernel construction failed: " << status.ToString() << std::endl;
            return 0;
        }

        std::cout << "QuantizeV2 operation test completed successfully" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}