#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract parameters from fuzzer input
        uint32_t input_size = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += sizeof(uint32_t);
        input_size = input_size % 100 + 1; // Limit size to reasonable range
        
        uint32_t dtype_idx = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += sizeof(uint32_t);
        dtype_idx = dtype_idx % 4; // Support 4 data types
        
        tensorflow::DataType dtype;
        switch (dtype_idx) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_DOUBLE; break;
            case 2: dtype = tensorflow::DT_HALF; break;
            default: dtype = tensorflow::DT_BFLOAT16; break;
        }
        
        bool signed_input = (*reinterpret_cast<const uint8_t*>(data + offset)) & 1;
        offset += sizeof(uint8_t);
        
        bool range_given = (*reinterpret_cast<const uint8_t*>(data + offset)) & 1;
        offset += sizeof(uint8_t);
        
        bool narrow_range = (*reinterpret_cast<const uint8_t*>(data + offset)) & 1;
        offset += sizeof(uint8_t);
        
        int32_t axis = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        axis = axis % 10 - 5; // Range from -5 to 4
        
        int32_t num_bits_val = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        num_bits_val = (num_bits_val % 16) + 1; // Range from 1 to 16
        
        if (offset + input_size * 4 > size) return 0;
        
        // Create TensorFlow scope and session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor
        tensorflow::TensorShape input_shape({static_cast<int64_t>(input_size)});
        tensorflow::Tensor input_tensor(dtype, input_shape);
        
        // Fill input tensor with fuzzer data
        if (dtype == tensorflow::DT_FLOAT) {
            auto input_flat = input_tensor.flat<float>();
            for (int i = 0; i < input_size && offset + sizeof(float) <= size; ++i) {
                float val = *reinterpret_cast<const float*>(data + offset);
                offset += sizeof(float);
                // Clamp to reasonable range to avoid overflow
                val = std::max(-1000.0f, std::min(1000.0f, val));
                input_flat(i) = val;
            }
        } else if (dtype == tensorflow::DT_DOUBLE) {
            auto input_flat = input_tensor.flat<double>();
            for (int i = 0; i < input_size && offset + sizeof(double) <= size; ++i) {
                double val = *reinterpret_cast<const double*>(data + offset);
                offset += sizeof(double);
                val = std::max(-1000.0, std::min(1000.0, val));
                input_flat(i) = val;
            }
        }
        
        // Create min/max tensors
        tensorflow::Tensor input_min_tensor(dtype, tensorflow::TensorShape({}));
        tensorflow::Tensor input_max_tensor(dtype, tensorflow::TensorShape({}));
        
        if (dtype == tensorflow::DT_FLOAT) {
            input_min_tensor.scalar<float>()() = -10.0f;
            input_max_tensor.scalar<float>()() = 10.0f;
        } else if (dtype == tensorflow::DT_DOUBLE) {
            input_min_tensor.scalar<double>()() = -10.0;
            input_max_tensor.scalar<double>()() = 10.0;
        }
        
        // Create num_bits tensor
        tensorflow::Tensor num_bits_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        num_bits_tensor.scalar<int32_t>()() = num_bits_val;
        
        // Create placeholder ops
        auto input_ph = tensorflow::ops::Placeholder(root, dtype);
        auto input_min_ph = tensorflow::ops::Placeholder(root, dtype);
        auto input_max_ph = tensorflow::ops::Placeholder(root, dtype);
        auto num_bits_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        
        // Create QuantizeAndDequantizeV3 operation
        auto quantize_op = tensorflow::ops::QuantizeAndDequantizeV3(
            root, input_ph, input_min_ph, input_max_ph, num_bits_ph,
            tensorflow::ops::QuantizeAndDequantizeV3::SignedInput(signed_input)
                .RangeGiven(range_given)
                .NarrowRange(narrow_range)
                .Axis(axis));
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{input_ph, input_tensor},
             {input_min_ph, input_min_tensor},
             {input_max_ph, input_max_tensor},
             {num_bits_ph, num_bits_tensor}},
            {quantize_op}, &outputs);
        
        if (!status.ok()) {
            // Operation failed, but this is expected for some invalid inputs
            return 0;
        }
        
        // Verify output tensor has same shape and type as input
        if (!outputs.empty()) {
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() != dtype || output.shape() != input_shape) {
                std::cout << "Output tensor properties mismatch" << std::endl;
                return -1;
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}