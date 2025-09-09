#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions and parameters from fuzz input
        uint32_t data_rows = (data[offset] % 10) + 1;
        offset++;
        uint32_t data_cols = (data[offset] % 10) + 1;
        offset++;
        uint32_t indices_size = (data[offset] % data_rows) + 1;
        offset++;
        uint32_t num_segments_val = (data[offset] % 5) + 1;
        offset++;
        bool sparse_gradient = data[offset] % 2;
        offset++;
        
        if (offset + data_rows * data_cols * 4 + indices_size * 4 + indices_size * 4 > size) {
            return 0;
        }
        
        // Create TensorFlow scope and session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create data tensor (float32)
        tensorflow::Tensor data_tensor(tensorflow::DT_FLOAT, 
            tensorflow::TensorShape({static_cast<int64_t>(data_rows), static_cast<int64_t>(data_cols)}));
        auto data_flat = data_tensor.flat<float>();
        for (int i = 0; i < data_rows * data_cols; i++) {
            if (offset + 4 <= size) {
                float val;
                memcpy(&val, data + offset, sizeof(float));
                data_flat(i) = val;
                offset += 4;
            } else {
                data_flat(i) = 0.0f;
            }
        }
        
        // Create indices tensor (int32)
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, 
            tensorflow::TensorShape({static_cast<int64_t>(indices_size)}));
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < indices_size; i++) {
            if (offset + 4 <= size) {
                int32_t val;
                memcpy(&val, data + offset, sizeof(int32_t));
                indices_flat(i) = abs(val) % data_rows;
                offset += 4;
            } else {
                indices_flat(i) = 0;
            }
        }
        
        // Create segment_ids tensor (int32)
        tensorflow::Tensor segment_ids_tensor(tensorflow::DT_INT32, 
            tensorflow::TensorShape({static_cast<int64_t>(indices_size)}));
        auto segment_ids_flat = segment_ids_tensor.flat<int32_t>();
        for (int i = 0; i < indices_size; i++) {
            if (offset + 4 <= size) {
                int32_t val;
                memcpy(&val, data + offset, sizeof(int32_t));
                segment_ids_flat(i) = abs(val) % num_segments_val;
                offset += 4;
            } else {
                segment_ids_flat(i) = 0;
            }
        }
        
        // Create num_segments tensor (int32)
        tensorflow::Tensor num_segments_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        num_segments_tensor.scalar<int32_t>()() = num_segments_val;
        
        // Create placeholder ops
        auto data_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto indices_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto segment_ids_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto num_segments_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        
        // Create SparseSegmentMeanWithNumSegments operation
        auto sparse_segment_mean = tensorflow::ops::SparseSegmentMeanWithNumSegments(
            root, 
            data_placeholder, 
            indices_placeholder, 
            segment_ids_placeholder, 
            num_segments_placeholder,
            tensorflow::ops::SparseSegmentMeanWithNumSegments::Attrs().SparseGradient(sparse_gradient)
        );
        
        // Create session and run
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        if (session == nullptr) {
            return 0;
        }
        
        TF_CHECK_OK(session->Create(root.graph()));
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {data_placeholder.node()->name(), data_tensor},
            {indices_placeholder.node()->name(), indices_tensor},
            {segment_ids_placeholder.node()->name(), segment_ids_tensor},
            {num_segments_placeholder.node()->name(), num_segments_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session->Run(inputs, {sparse_segment_mean.node()->name()}, {}, &outputs);
        
        if (!status.ok()) {
            // Operation failed, which is acceptable for fuzzing
            return 0;
        }
        
        // Verify output tensor properties
        if (!outputs.empty()) {
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() == tensorflow::DT_FLOAT && 
                output.shape().dims() == 2 &&
                output.shape().dim_size(0) == num_segments_val &&
                output.shape().dim_size(1) == data_cols) {
                // Output has expected shape and type
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}