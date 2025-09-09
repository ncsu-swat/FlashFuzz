#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/cc/framework/scope.h>

constexpr uint8_t MIN_RANK = 0;
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

        tensorflow::DataType input_dtype = parseDataType(data[offset++]);
        uint8_t input_rank = parseRank(data[offset++]);
        std::vector<int64_t> input_shape = parseShape(data, offset, size, input_rank);
        
        tensorflow::TensorShape input_tensor_shape(input_shape);
        tensorflow::Tensor input_tensor(input_dtype, input_tensor_shape);
        fillTensorWithDataByType(input_tensor, input_dtype, data, offset, size);

        tensorflow::TensorShape scalar_shape({});
        tensorflow::Tensor input_min_tensor(input_dtype, scalar_shape);
        fillTensorWithDataByType(input_min_tensor, input_dtype, data, offset, size);

        tensorflow::Tensor input_max_tensor(input_dtype, scalar_shape);
        fillTensorWithDataByType(input_max_tensor, input_dtype, data, offset, size);

        tensorflow::Tensor num_bits_tensor(tensorflow::DT_INT32, scalar_shape);
        if (offset + sizeof(int32_t) <= size) {
            int32_t num_bits_val;
            std::memcpy(&num_bits_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            num_bits_val = std::abs(num_bits_val) % 16 + 1;
            num_bits_tensor.scalar<int32_t>()() = num_bits_val;
        } else {
            num_bits_tensor.scalar<int32_t>()() = 8;
        }

        bool signed_input = true;
        bool range_given = true;
        bool narrow_range = false;
        int axis = -1;

        if (offset < size) {
            signed_input = (data[offset++] % 2) == 1;
        }
        if (offset < size) {
            range_given = (data[offset++] % 2) == 1;
        }
        if (offset < size) {
            narrow_range = (data[offset++] % 2) == 1;
        }
        if (offset + sizeof(int32_t) <= size) {
            int32_t axis_val;
            std::memcpy(&axis_val, data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            axis = axis_val % (input_rank + 1) - 1;
        }

        std::cout << "Input tensor shape: ";
        for (int i = 0; i < input_tensor_shape.dims(); ++i) {
            std::cout << input_tensor_shape.dim_size(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Input dtype: " << tensorflow::DataTypeString(input_dtype) << std::endl;
        std::cout << "Num bits: " << num_bits_tensor.scalar<int32_t>()() << std::endl;
        std::cout << "Signed input: " << signed_input << std::endl;
        std::cout << "Range given: " << range_given << std::endl;
        std::cout << "Narrow range: " << narrow_range << std::endl;
        std::cout << "Axis: " << axis << std::endl;

        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto input_placeholder = tensorflow::ops::Placeholder(root, input_dtype);
        auto input_min_placeholder = tensorflow::ops::Placeholder(root, input_dtype);
        auto input_max_placeholder = tensorflow::ops::Placeholder(root, input_dtype);
        auto num_bits_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);

        auto quantize_op = tensorflow::ops::QuantizeAndDequantizeV3(
            root,
            input_placeholder,
            input_min_placeholder,
            input_max_placeholder,
            num_bits_placeholder,
            tensorflow::ops::QuantizeAndDequantizeV3::SignedInput(signed_input)
                .RangeGiven(range_given)
                .NarrowRange(narrow_range)
                .Axis(axis)
        );

        tensorflow::GraphDef graph;
        TF_CHECK_OK(root.ToGraphDef(&graph));

        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        TF_CHECK_OK(session->Create(graph));

        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {input_placeholder.node()->name(), input_tensor},
            {input_min_placeholder.node()->name(), input_min_tensor},
            {input_max_placeholder.node()->name(), input_max_tensor},
            {num_bits_placeholder.node()->name(), num_bits_tensor}
        };

        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session->Run(inputs, {quantize_op.node()->name()}, {}, &outputs);

        if (status.ok() && !outputs.empty()) {
            std::cout << "Operation completed successfully" << std::endl;
            std::cout << "Output tensor shape: ";
            for (int i = 0; i < outputs[0].shape().dims(); ++i) {
                std::cout << outputs[0].shape().dim_size(i) << " ";
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