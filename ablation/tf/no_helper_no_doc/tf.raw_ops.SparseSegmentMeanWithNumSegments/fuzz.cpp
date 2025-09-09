#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/sparse_segment_reduction_ops.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/str_util.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/node_def_util.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/kernels/ops_util.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions from fuzz input
        uint32_t data_rows = (data[offset] % 10) + 1;
        offset++;
        uint32_t data_cols = (data[offset] % 10) + 1;
        offset++;
        uint32_t indices_size = (data[offset] % data_rows) + 1;
        offset++;
        uint32_t segment_ids_size = indices_size;
        uint32_t num_segments = (data[offset] % 10) + 1;
        offset++;
        
        if (offset + data_rows * data_cols * sizeof(float) + 
            indices_size * sizeof(int32_t) + 
            segment_ids_size * sizeof(int32_t) + 
            sizeof(int32_t) > size) {
            return 0;
        }
        
        // Create data tensor
        tensorflow::Tensor data_tensor(tensorflow::DT_FLOAT, 
                                     tensorflow::TensorShape({static_cast<int64_t>(data_rows), 
                                                             static_cast<int64_t>(data_cols)}));
        auto data_flat = data_tensor.flat<float>();
        for (int i = 0; i < data_rows * data_cols && offset + sizeof(float) <= size; i++) {
            float val;
            memcpy(&val, data + offset, sizeof(float));
            data_flat(i) = val;
            offset += sizeof(float);
        }
        
        // Create indices tensor
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, 
                                        tensorflow::TensorShape({static_cast<int64_t>(indices_size)}));
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < indices_size && offset + sizeof(int32_t) <= size; i++) {
            int32_t val;
            memcpy(&val, data + offset, sizeof(int32_t));
            indices_flat(i) = abs(val) % data_rows;
            offset += sizeof(int32_t);
        }
        
        // Create segment_ids tensor
        tensorflow::Tensor segment_ids_tensor(tensorflow::DT_INT32, 
                                             tensorflow::TensorShape({static_cast<int64_t>(segment_ids_size)}));
        auto segment_ids_flat = segment_ids_tensor.flat<int32_t>();
        int32_t current_segment = 0;
        for (int i = 0; i < segment_ids_size; i++) {
            if (i > 0 && offset < size && (data[offset % size] % 3) == 0) {
                current_segment = std::min(current_segment + 1, static_cast<int32_t>(num_segments - 1));
            }
            segment_ids_flat(i) = current_segment;
            if (offset < size) offset++;
        }
        
        // Create num_segments tensor
        tensorflow::Tensor num_segments_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        num_segments_tensor.scalar<int32_t>()() = static_cast<int32_t>(num_segments);
        
        // Create output tensor
        tensorflow::Tensor output_tensor(tensorflow::DT_FLOAT, 
                                       tensorflow::TensorShape({static_cast<int64_t>(num_segments), 
                                                               static_cast<int64_t>(data_cols)}));
        
        // Create a simple computation context
        tensorflow::OpKernelContext::Params params;
        tensorflow::DeviceBase device(tensorflow::Env::Default());
        params.device = &device;
        params.step_id = 1;
        
        // Prepare input tensors
        std::vector<tensorflow::Tensor> inputs = {data_tensor, indices_tensor, segment_ids_tensor, num_segments_tensor};
        std::vector<tensorflow::Tensor*> input_ptrs;
        for (auto& tensor : inputs) {
            input_ptrs.push_back(&tensor);
        }
        
        // Simple validation of the operation logic
        auto data_matrix = data_tensor.matrix<float>();
        auto indices_vec = indices_tensor.vec<int32_t>();
        auto segment_ids_vec = segment_ids_tensor.vec<int32_t>();
        auto output_matrix = output_tensor.matrix<float>();
        
        // Initialize output to zero
        output_matrix.setZero();
        
        // Count elements per segment
        std::vector<int32_t> segment_counts(num_segments, 0);
        for (int i = 0; i < segment_ids_size; i++) {
            int32_t segment_id = segment_ids_vec(i);
            if (segment_id >= 0 && segment_id < num_segments) {
                segment_counts[segment_id]++;
            }
        }
        
        // Compute segment sums
        for (int i = 0; i < indices_size; i++) {
            int32_t data_idx = indices_vec(i);
            int32_t segment_id = segment_ids_vec(i);
            
            if (data_idx >= 0 && data_idx < data_rows && 
                segment_id >= 0 && segment_id < num_segments) {
                for (int j = 0; j < data_cols; j++) {
                    output_matrix(segment_id, j) += data_matrix(data_idx, j);
                }
            }
        }
        
        // Compute means
        for (int i = 0; i < num_segments; i++) {
            if (segment_counts[i] > 0) {
                for (int j = 0; j < data_cols; j++) {
                    output_matrix(i, j) /= static_cast<float>(segment_counts[i]);
                }
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}