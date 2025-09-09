#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/sparse_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract parameters from fuzzer input
        uint32_t num_indices = (data[offset] % 10) + 1;
        offset += 1;
        
        uint32_t num_values = (data[offset] % 10) + 1;
        offset += 1;
        
        uint32_t dense_shape_val = (data[offset] % 100) + 1;
        offset += 1;
        
        uint32_t size_val = (data[offset] % 50) + 1;
        offset += 1;
        
        uint32_t num_weights = data[offset] % 2 == 0 ? 0 : num_values;
        offset += 1;
        
        bool binary_output = data[offset] % 2 == 1;
        offset += 1;
        
        // Determine data types from fuzzer input
        tensorflow::DataType values_type = (data[offset] % 2 == 0) ? tensorflow::DT_INT32 : tensorflow::DT_INT64;
        offset += 1;
        
        tensorflow::DataType weights_type;
        switch (data[offset] % 4) {
            case 0: weights_type = tensorflow::DT_INT32; break;
            case 1: weights_type = tensorflow::DT_INT64; break;
            case 2: weights_type = tensorflow::DT_FLOAT; break;
            default: weights_type = tensorflow::DT_DOUBLE; break;
        }
        offset += 1;
        
        if (offset + num_indices * 2 * 8 + num_values * 8 + num_weights * 8 > size) {
            return 0;
        }
        
        // Create scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create indices tensor (2D int64)
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({static_cast<int64_t>(num_indices), 2}));
        auto indices_flat = indices_tensor.flat<int64_t>();
        for (uint32_t i = 0; i < num_indices * 2; ++i) {
            if (offset + 8 <= size) {
                int64_t val;
                memcpy(&val, data + offset, sizeof(int64_t));
                indices_flat(i) = std::abs(val) % dense_shape_val;
                offset += 8;
            } else {
                indices_flat(i) = i % dense_shape_val;
            }
        }
        
        // Create values tensor
        tensorflow::Tensor values_tensor;
        if (values_type == tensorflow::DT_INT32) {
            values_tensor = tensorflow::Tensor(tensorflow::DT_INT32, tensorflow::TensorShape({static_cast<int64_t>(num_values)}));
            auto values_flat = values_tensor.flat<int32_t>();
            for (uint32_t i = 0; i < num_values; ++i) {
                if (offset + 4 <= size) {
                    int32_t val;
                    memcpy(&val, data + offset, sizeof(int32_t));
                    values_flat(i) = std::abs(val) % size_val;
                    offset += 4;
                } else {
                    values_flat(i) = i % size_val;
                }
            }
        } else {
            values_tensor = tensorflow::Tensor(tensorflow::DT_INT64, tensorflow::TensorShape({static_cast<int64_t>(num_values)}));
            auto values_flat = values_tensor.flat<int64_t>();
            for (uint32_t i = 0; i < num_values; ++i) {
                if (offset + 8 <= size) {
                    int64_t val;
                    memcpy(&val, data + offset, sizeof(int64_t));
                    values_flat(i) = std::abs(val) % size_val;
                    offset += 8;
                } else {
                    values_flat(i) = i % size_val;
                }
            }
        }
        
        // Create dense_shape tensor
        tensorflow::Tensor dense_shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({1}));
        dense_shape_tensor.flat<int64_t>()(0) = dense_shape_val;
        
        // Create size tensor
        tensorflow::Tensor size_tensor;
        if (values_type == tensorflow::DT_INT32) {
            size_tensor = tensorflow::Tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
            size_tensor.scalar<int32_t>()() = static_cast<int32_t>(size_val);
        } else {
            size_tensor = tensorflow::Tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
            size_tensor.scalar<int64_t>()() = static_cast<int64_t>(size_val);
        }
        
        // Create weights tensor
        tensorflow::Tensor weights_tensor;
        if (num_weights == 0) {
            weights_tensor = tensorflow::Tensor(weights_type, tensorflow::TensorShape({0}));
        } else {
            weights_tensor = tensorflow::Tensor(weights_type, tensorflow::TensorShape({static_cast<int64_t>(num_weights)}));
            
            switch (weights_type) {
                case tensorflow::DT_INT32: {
                    auto weights_flat = weights_tensor.flat<int32_t>();
                    for (uint32_t i = 0; i < num_weights; ++i) {
                        if (offset + 4 <= size) {
                            int32_t val;
                            memcpy(&val, data + offset, sizeof(int32_t));
                            weights_flat(i) = val;
                            offset += 4;
                        } else {
                            weights_flat(i) = 1;
                        }
                    }
                    break;
                }
                case tensorflow::DT_INT64: {
                    auto weights_flat = weights_tensor.flat<int64_t>();
                    for (uint32_t i = 0; i < num_weights; ++i) {
                        if (offset + 8 <= size) {
                            int64_t val;
                            memcpy(&val, data + offset, sizeof(int64_t));
                            weights_flat(i) = val;
                            offset += 8;
                        } else {
                            weights_flat(i) = 1;
                        }
                    }
                    break;
                }
                case tensorflow::DT_FLOAT: {
                    auto weights_flat = weights_tensor.flat<float>();
                    for (uint32_t i = 0; i < num_weights; ++i) {
                        if (offset + 4 <= size) {
                            float val;
                            memcpy(&val, data + offset, sizeof(float));
                            weights_flat(i) = std::isfinite(val) ? val : 1.0f;
                            offset += 4;
                        } else {
                            weights_flat(i) = 1.0f;
                        }
                    }
                    break;
                }
                case tensorflow::DT_DOUBLE: {
                    auto weights_flat = weights_tensor.flat<double>();
                    for (uint32_t i = 0; i < num_weights; ++i) {
                        if (offset + 8 <= size) {
                            double val;
                            memcpy(&val, data + offset, sizeof(double));
                            weights_flat(i) = std::isfinite(val) ? val : 1.0;
                            offset += 8;
                        } else {
                            weights_flat(i) = 1.0;
                        }
                    }
                    break;
                }
            }
        }
        
        // Create input nodes
        auto indices_input = tensorflow::ops::Const(root, indices_tensor);
        auto values_input = tensorflow::ops::Const(root, values_tensor);
        auto dense_shape_input = tensorflow::ops::Const(root, dense_shape_tensor);
        auto size_input = tensorflow::ops::Const(root, size_tensor);
        auto weights_input = tensorflow::ops::Const(root, weights_tensor);
        
        // Create SparseBincount operation
        auto sparse_bincount = tensorflow::ops::SparseBincount(
            root,
            indices_input,
            values_input,
            dense_shape_input,
            size_input,
            weights_input,
            tensorflow::ops::SparseBincount::BinaryOutput(binary_output)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({sparse_bincount}, &outputs);
        
        if (!status.ok()) {
            std::cout << "SparseBincount operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}