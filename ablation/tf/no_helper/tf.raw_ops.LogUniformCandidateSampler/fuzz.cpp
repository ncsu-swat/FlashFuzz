#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/candidate_sampling_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 20) return 0;
        
        // Extract parameters from fuzzer input
        int32_t batch_size = (data[offset] % 10) + 1;
        offset++;
        
        int32_t num_true = (data[offset] % 5) + 1;
        offset++;
        
        int32_t num_sampled = (data[offset] % 100) + 1;
        offset++;
        
        bool unique = data[offset] % 2;
        offset++;
        
        int32_t range_max = ((data[offset] << 8) | data[offset + 1]) % 10000 + 1;
        offset += 2;
        
        int32_t seed = (data[offset] << 24) | (data[offset + 1] << 16) | 
                       (data[offset + 2] << 8) | data[offset + 3];
        offset += 4;
        
        int32_t seed2 = (data[offset] << 24) | (data[offset + 1] << 16) | 
                        (data[offset + 2] << 8) | data[offset + 3];
        offset += 4;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create true_classes tensor
        tensorflow::TensorShape true_classes_shape({batch_size, num_true});
        tensorflow::Tensor true_classes_tensor(tensorflow::DT_INT64, true_classes_shape);
        auto true_classes_flat = true_classes_tensor.flat<int64_t>();
        
        // Fill true_classes with data from fuzzer input
        for (int i = 0; i < batch_size * num_true && offset < size; i++) {
            true_classes_flat(i) = (data[offset] % range_max);
            offset++;
        }
        
        // Create placeholder for true_classes
        auto true_classes_placeholder = tensorflow::ops::Placeholder(
            root.WithOpName("true_classes"), tensorflow::DT_INT64,
            tensorflow::ops::Placeholder::Shape(true_classes_shape));
        
        // Create LogUniformCandidateSampler operation
        auto sampler = tensorflow::ops::LogUniformCandidateSampler(
            root.WithOpName("log_uniform_sampler"),
            true_classes_placeholder,
            num_true,
            num_sampled,
            unique,
            range_max,
            tensorflow::ops::LogUniformCandidateSampler::Seed(seed),
            tensorflow::ops::LogUniformCandidateSampler::Seed2(seed2)
        );
        
        // Create session and run the operation
        tensorflow::ClientSession session(root);
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run(
            {{true_classes_placeholder, true_classes_tensor}},
            {sampler.sampled_candidates, sampler.true_expected_count, sampler.sampled_expected_count},
            &outputs
        );
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Validate outputs
        if (outputs.size() != 3) {
            std::cout << "Expected 3 outputs, got " << outputs.size() << std::endl;
            return 0;
        }
        
        // Check sampled_candidates shape and type
        if (outputs[0].dtype() != tensorflow::DT_INT64) {
            std::cout << "sampled_candidates has wrong dtype" << std::endl;
            return 0;
        }
        
        if (outputs[0].shape().dims() != 1 || outputs[0].shape().dim_size(0) != num_sampled) {
            std::cout << "sampled_candidates has wrong shape" << std::endl;
            return 0;
        }
        
        // Check true_expected_count shape and type
        if (outputs[1].dtype() != tensorflow::DT_FLOAT) {
            std::cout << "true_expected_count has wrong dtype" << std::endl;
            return 0;
        }
        
        if (outputs[1].shape().dims() != 2 || 
            outputs[1].shape().dim_size(0) != batch_size ||
            outputs[1].shape().dim_size(1) != num_true) {
            std::cout << "true_expected_count has wrong shape" << std::endl;
            return 0;
        }
        
        // Check sampled_expected_count shape and type
        if (outputs[2].dtype() != tensorflow::DT_FLOAT) {
            std::cout << "sampled_expected_count has wrong dtype" << std::endl;
            return 0;
        }
        
        if (outputs[2].shape().dims() != 1 || outputs[2].shape().dim_size(0) != num_sampled) {
            std::cout << "sampled_expected_count has wrong shape" << std::endl;
            return 0;
        }
        
        // Verify sampled candidates are within range
        auto sampled_flat = outputs[0].flat<int64_t>();
        for (int i = 0; i < num_sampled; i++) {
            if (sampled_flat(i) < 0 || sampled_flat(i) >= range_max) {
                std::cout << "sampled candidate out of range: " << sampled_flat(i) << std::endl;
                return 0;
            }
        }
        
        // If unique is true, check for duplicates
        if (unique) {
            for (int i = 0; i < num_sampled; i++) {
                for (int j = i + 1; j < num_sampled; j++) {
                    if (sampled_flat(i) == sampled_flat(j)) {
                        std::cout << "Duplicate candidates found when unique=true" << std::endl;
                        return 0;
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