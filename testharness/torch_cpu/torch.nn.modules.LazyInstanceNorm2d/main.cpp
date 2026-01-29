#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions for InstanceNorm2d (N, C, H, W)
        if (input.dim() < 4) {
            std::vector<int64_t> new_shape;
            for (int64_t i = 0; i < input.dim(); i++) {
                new_shape.push_back(input.size(i));
            }
            while (new_shape.size() < 4) {
                new_shape.push_back(1);
            }
            input = input.reshape(new_shape);
        }
        
        // Ensure input is float type for normalization
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Get num_features from channel dimension (dimension 1)
        int64_t num_features = input.size(1);
        if (num_features <= 0) {
            return 0;
        }
        
        // Extract parameters for InstanceNorm2d
        uint8_t param_byte = 0;
        if (offset < Size) {
            param_byte = Data[offset++];
        }
        
        // Parse eps parameter (small value to avoid division by zero)
        double eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            float eps_f;
            std::memcpy(&eps_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            eps_f = std::abs(eps_f);
            if (std::isfinite(eps_f) && eps_f > 0.0f && eps_f < 1.0f) {
                eps = static_cast<double>(eps_f);
            }
        }
        
        // Parse momentum parameter
        double momentum = 0.1;
        if (offset + sizeof(float) <= Size) {
            float mom_f;
            std::memcpy(&mom_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            mom_f = std::abs(mom_f);
            if (std::isfinite(mom_f)) {
                momentum = std::fmod(static_cast<double>(mom_f), 1.0);
                if (momentum <= 0.0) momentum = 0.1;
            }
        }
        
        // Parse affine parameter
        bool affine = (param_byte & 0x01) != 0;
        
        // Parse track_running_stats parameter
        bool track_running_stats = (param_byte & 0x02) != 0;
        
        // Create InstanceNorm2d module with num_features derived from input
        torch::nn::InstanceNorm2d instance_norm(
            torch::nn::InstanceNorm2dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply the module to the input tensor
        torch::Tensor output = instance_norm(input);
        
        // Force computation to ensure any potential errors are triggered
        output = output.contiguous();
        
        // Access some elements to ensure computation is performed
        if (output.numel() > 0) {
            float sum = output.sum().item<float>();
            (void)sum;
        }
        
        // Test with a second input to exercise the module after initialization
        if (offset + 4 <= Size) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input2.dim() < 4) {
                std::vector<int64_t> new_shape2;
                for (int64_t i = 0; i < input2.dim(); i++) {
                    new_shape2.push_back(input2.size(i));
                }
                while (new_shape2.size() < 4) {
                    new_shape2.push_back(1);
                }
                input2 = input2.reshape(new_shape2);
            }
            if (!input2.is_floating_point()) {
                input2 = input2.to(torch::kFloat32);
            }
            
            // Adjust channel dimension to match the initialized module
            if (input2.size(1) != num_features) {
                std::vector<int64_t> adj_shape = {input2.size(0), num_features, input2.size(2), input2.size(3)};
                int64_t total_elements = input2.numel();
                int64_t needed_elements = 1;
                for (auto s : adj_shape) needed_elements *= s;
                if (total_elements >= needed_elements && needed_elements > 0) {
                    input2 = input2.flatten().slice(0, 0, needed_elements).reshape(adj_shape);
                    try {
                        torch::Tensor output2 = instance_norm(input2);
                        output2 = output2.contiguous();
                    } catch (...) {
                        // Shape mismatch is expected in some cases
                    }
                }
            } else {
                try {
                    torch::Tensor output2 = instance_norm(input2);
                    output2 = output2.contiguous();
                } catch (...) {
                    // Expected for shape mismatches
                }
            }
        }
        
        // Test eval mode
        instance_norm->eval();
        try {
            torch::Tensor output_eval = instance_norm(input);
            output_eval = output_eval.contiguous();
        } catch (...) {
            // May fail in eval mode with certain configurations
        }
        
        // Test train mode again
        instance_norm->train();
        try {
            torch::Tensor output_train = instance_norm(input);
            output_train = output_train.contiguous();
        } catch (...) {
            // Expected in some cases
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}