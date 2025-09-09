#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions for spectrogram tensor
        uint32_t batch_size = *reinterpret_cast<const uint32_t*>(data + offset) % 8 + 1;
        offset += 4;
        uint32_t time_steps = *reinterpret_cast<const uint32_t*>(data + offset) % 256 + 1;
        offset += 4;
        uint32_t freq_bins = *reinterpret_cast<const uint32_t*>(data + offset) % 256 + 1;
        offset += 4;
        
        // Extract sample rate
        int32_t sample_rate = *reinterpret_cast<const int32_t*>(data + offset);
        if (sample_rate <= 0) sample_rate = 16000;
        offset += 4;
        
        // Extract optional parameters
        float upper_freq = 4000.0f;
        float lower_freq = 20.0f;
        int filterbank_channels = 40;
        int dct_coeffs = 13;
        
        if (offset + 16 <= size) {
            upper_freq = *reinterpret_cast<const float*>(data + offset);
            if (upper_freq <= 0 || upper_freq > 22050) upper_freq = 4000.0f;
            offset += 4;
            
            lower_freq = *reinterpret_cast<const float*>(data + offset);
            if (lower_freq < 0 || lower_freq >= upper_freq) lower_freq = 20.0f;
            offset += 4;
            
            filterbank_channels = *reinterpret_cast<const int32_t*>(data + offset) % 128 + 1;
            offset += 4;
            
            dct_coeffs = *reinterpret_cast<const int32_t*>(data + offset) % 64 + 1;
            offset += 4;
        }
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create spectrogram tensor with random data
        tensorflow::TensorShape spec_shape({static_cast<int64_t>(batch_size), 
                                           static_cast<int64_t>(time_steps), 
                                           static_cast<int64_t>(freq_bins)});
        tensorflow::Tensor spectrogram_tensor(tensorflow::DT_FLOAT, spec_shape);
        auto spec_flat = spectrogram_tensor.flat<float>();
        
        // Fill with data from fuzzer input
        size_t spec_size = batch_size * time_steps * freq_bins;
        for (size_t i = 0; i < spec_size && offset + 4 <= size; ++i) {
            float val = *reinterpret_cast<const float*>(data + offset);
            if (std::isnan(val) || std::isinf(val)) val = 0.0f;
            spec_flat(i) = std::abs(val); // Spectrogram values should be non-negative
            offset += 4;
            if (offset >= size) break;
        }
        
        // Fill remaining values with small positive numbers
        for (size_t i = offset/4; i < spec_size; ++i) {
            spec_flat(i) = 0.01f;
        }
        
        // Create sample rate tensor
        tensorflow::Tensor sample_rate_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        sample_rate_tensor.scalar<int32_t>()() = sample_rate;
        
        // Create constant ops
        auto spectrogram_op = tensorflow::ops::Const(root, spectrogram_tensor);
        auto sample_rate_op = tensorflow::ops::Const(root, sample_rate_tensor);
        
        // Create MFCC operation using raw ops
        tensorflow::Node* mfcc_node;
        tensorflow::NodeBuilder builder("mfcc", "Mfcc");
        builder.Input(spectrogram_op.node())
               .Input(sample_rate_op.node())
               .Attr("upper_frequency_limit", upper_freq)
               .Attr("lower_frequency_limit", lower_freq)
               .Attr("filterbank_channel_count", filterbank_channels)
               .Attr("dct_coefficient_count", dct_coeffs);
        
        tensorflow::Status status = builder.Finalize(root.graph(), &mfcc_node);
        if (!status.ok()) {
            return 0;
        }
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        status = session.Run({tensorflow::Output(mfcc_node)}, &outputs);
        if (!status.ok()) {
            return 0;
        }
        
        // Verify output shape and values
        if (!outputs.empty()) {
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() == tensorflow::DT_FLOAT) {
                auto output_flat = output.flat<float>();
                for (int i = 0; i < output_flat.size(); ++i) {
                    float val = output_flat(i);
                    if (std::isnan(val) || std::isinf(val)) {
                        break;
                    }
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