#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/kernels/ops_testutil.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract parameters from fuzzer input
        int32_t sample_rate = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        float upper_frequency_limit = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        float lower_frequency_limit = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        int32_t filterbank_channel_count = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        // Clamp values to reasonable ranges
        sample_rate = std::max(1000, std::min(48000, sample_rate));
        upper_frequency_limit = std::max(100.0f, std::min(8000.0f, std::abs(upper_frequency_limit)));
        lower_frequency_limit = std::max(20.0f, std::min(upper_frequency_limit - 1.0f, std::abs(lower_frequency_limit)));
        filterbank_channel_count = std::max(1, std::min(40, filterbank_channel_count));
        
        // Calculate remaining data for spectrogram tensor
        size_t remaining_size = size - offset;
        if (remaining_size < sizeof(float)) return 0;
        
        // Create dimensions for spectrogram tensor
        int batch_size = 1;
        int time_frames = std::max(1, static_cast<int>(remaining_size / (sizeof(float) * 129)));
        int freq_bins = 129; // Common FFT size for audio processing
        
        // Ensure we don't exceed available data
        size_t required_elements = batch_size * time_frames * freq_bins;
        size_t available_elements = remaining_size / sizeof(float);
        if (required_elements > available_elements) {
            time_frames = std::max(1, static_cast<int>(available_elements / freq_bins));
            required_elements = batch_size * time_frames * freq_bins;
        }
        
        // Create input tensor (spectrogram)
        tensorflow::Tensor spectrogram_tensor(tensorflow::DT_FLOAT, 
            tensorflow::TensorShape({batch_size, time_frames, freq_bins}));
        
        auto spectrogram_flat = spectrogram_tensor.flat<float>();
        
        // Fill tensor with fuzzer data
        for (int i = 0; i < required_elements && offset + sizeof(float) <= size; ++i) {
            if (offset + sizeof(float) <= size) {
                spectrogram_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += sizeof(float);
            } else {
                spectrogram_flat(i) = 0.0f;
            }
        }
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Build graph with Mfcc operation
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* spectrogram_node = graph_def.add_node();
        spectrogram_node->set_name("spectrogram");
        spectrogram_node->set_op("Placeholder");
        (*spectrogram_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        (*spectrogram_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(batch_size);
        (*spectrogram_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(time_frames);
        (*spectrogram_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(freq_bins);
        
        tensorflow::NodeDef* sample_rate_node = graph_def.add_node();
        sample_rate_node->set_name("sample_rate");
        sample_rate_node->set_op("Const");
        (*sample_rate_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        tensorflow::TensorProto* sample_rate_proto = (*sample_rate_node->mutable_attr())["value"].mutable_tensor();
        sample_rate_proto->set_dtype(tensorflow::DT_INT32);
        sample_rate_proto->add_int_val(sample_rate);
        
        tensorflow::NodeDef* mfcc_node = graph_def.add_node();
        mfcc_node->set_name("mfcc");
        mfcc_node->set_op("Mfcc");
        mfcc_node->add_input("spectrogram");
        mfcc_node->add_input("sample_rate");
        (*mfcc_node->mutable_attr())["upper_frequency_limit"].set_f(upper_frequency_limit);
        (*mfcc_node->mutable_attr())["lower_frequency_limit"].set_f(lower_frequency_limit);
        (*mfcc_node->mutable_attr())["filterbank_channel_count"].set_i(filterbank_channel_count);
        (*mfcc_node->mutable_attr())["dct_coefficient_count"].set_i(13);
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Prepare inputs
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"spectrogram", spectrogram_tensor}
        };
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"mfcc"}, {}, &outputs);
        
        // Clean up
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}