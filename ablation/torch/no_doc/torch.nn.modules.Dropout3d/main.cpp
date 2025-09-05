#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 4) {
            // Need minimum bytes for basic parameters
            return 0;
        }

        size_t offset = 0;

        // Parse dropout probability from first byte
        double p = static_cast<double>(Data[offset++]) / 255.0;
        
        // Parse inplace flag
        bool inplace = (Data[offset++] % 2) == 1;
        
        // Parse training mode flag
        bool training_mode = (Data[offset++] % 2) == 1;
        
        // Create Dropout3d module with parsed parameters
        torch::nn::Dropout3dOptions options(p);
        options.inplace(inplace);
        torch::nn::Dropout3d dropout3d(options);
        
        // Set training/eval mode
        if (training_mode) {
            dropout3d->train();
        } else {
            dropout3d->eval();
        }
        
        // Try to create input tensor from remaining data
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, create a random 5D tensor
            uint8_t rank_selector = Data[offset % Size];
            int dims[5];
            for (int i = 0; i < 5; i++) {
                if (offset + i < Size) {
                    dims[i] = 1 + (Data[offset + i] % 8);
                } else {
                    dims[i] = 2;
                }
            }
            
            // Dropout3d expects 5D input: (N, C, D, H, W)
            input = torch::randn({dims[0], dims[1], dims[2], dims[3], dims[4]});
        }
        
        // Test various tensor shapes and configurations
        std::vector<torch::Tensor> test_tensors;
        
        // Add the parsed/created tensor
        test_tensors.push_back(input);
        
        // Add edge case tensors based on fuzzer data
        if (Size > offset) {
            uint8_t edge_selector = Data[offset % Size];
            
            // 5D tensor with minimum dimensions
            if (edge_selector & 0x01) {
                test_tensors.push_back(torch::randn({1, 1, 1, 1, 1}));
            }
            
            // 5D tensor with zero dimension (should fail gracefully)
            if (edge_selector & 0x02) {
                test_tensors.push_back(torch::randn({0, 1, 1, 1, 1}));
            }
            
            // Large channel dimension
            if (edge_selector & 0x04) {
                test_tensors.push_back(torch::randn({2, 64, 2, 2, 2}));
            }
            
            // Non-contiguous tensor
            if (edge_selector & 0x08) {
                auto t = torch::randn({2, 3, 4, 4, 4});
                test_tensors.push_back(t.transpose(1, 2));
            }
            
            // Different dtypes
            if (edge_selector & 0x10) {
                test_tensors.push_back(torch::randn({2, 3, 2, 2, 2}, torch::kDouble));
            }
            
            if (edge_selector & 0x20) {
                test_tensors.push_back(torch::randn({2, 3, 2, 2, 2}, torch::kHalf));
            }
            
            // 3D tensor (should be rejected or handled)
            if (edge_selector & 0x40) {
                test_tensors.push_back(torch::randn({2, 3, 4}));
            }
            
            // 4D tensor (should be rejected or handled)
            if (edge_selector & 0x80) {
                test_tensors.push_back(torch::randn({2, 3, 4, 5}));
            }
        }
        
        // Test with requires_grad variations
        if (Size > offset + 1) {
            uint8_t grad_selector = Data[(offset + 1) % Size];
            if (grad_selector & 0x01) {
                auto t = torch::randn({2, 3, 2, 2, 2}, torch::requires_grad());
                test_tensors.push_back(t);
            }
        }
        
        // Process all test tensors
        for (auto& tensor : test_tensors) {
            try {
                // Forward pass
                torch::Tensor output = dropout3d->forward(tensor);
                
                // Verify output properties
                if (output.defined()) {
                    // Check shape preservation (except for potential inplace modification)
                    if (!inplace && output.sizes() != tensor.sizes()) {
                        std::cerr << "Shape mismatch: input " << tensor.sizes() 
                                  << " vs output " << output.sizes() << std::endl;
                    }
                    
                    // In eval mode, output should equal input
                    if (!training_mode && !torch::allclose(output, tensor, 1e-5, 1e-8)) {
                        std::cerr << "Eval mode: output doesn't match input" << std::endl;
                    }
                    
                    // In training mode with p=0, output should equal input
                    if (training_mode && p == 0.0 && !torch::allclose(output, tensor, 1e-5, 1e-8)) {
                        std::cerr << "Training with p=0: output doesn't match input" << std::endl;
                    }
                    
                    // Check for NaN or Inf
                    if (output.isnan().any().item<bool>()) {
                        std::cerr << "Output contains NaN" << std::endl;
                    }
                    if (output.isinf().any().item<bool>()) {
                        std::cerr << "Output contains Inf" << std::endl;
                    }
                    
                    // Test backward pass if tensor requires grad
                    if (tensor.requires_grad() && output.requires_grad()) {
                        try {
                            auto loss = output.sum();
                            loss.backward();
                        } catch (const c10::Error& e) {
                            // Backward pass failed, but continue
                        }
                    }
                }
                
            } catch (const c10::Error& e) {
                // PyTorch errors are expected for invalid inputs
                // Continue testing other tensors
            } catch (const std::exception& e) {
                // Other exceptions, log but continue
                std::cerr << "Processing error: " << e.what() << std::endl;
            }
        }
        
        // Additional edge case: multiple forward passes
        if (!test_tensors.empty()) {
            auto& first_tensor = test_tensors[0];
            if (first_tensor.dim() == 5) {  // Only for valid 5D tensors
                try {
                    // Multiple forward passes to test consistency
                    torch::Tensor out1 = dropout3d->forward(first_tensor);
                    torch::Tensor out2 = dropout3d->forward(first_tensor);
                    
                    // In eval mode, outputs should be identical
                    if (!training_mode && !torch::equal(out1, out2)) {
                        std::cerr << "Eval mode: inconsistent outputs across forward passes" << std::endl;
                    }
                } catch (const c10::Error& e) {
                    // Expected for invalid inputs
                }
            }
        }
        
        // Test state changes
        dropout3d->eval();
        dropout3d->train();
        
        // Test parameter access
        auto params = dropout3d->parameters();
        auto buffers = dropout3d->buffers();
        
        // Dropout3d should have no parameters or buffers
        if (!params.empty()) {
            std::cerr << "Unexpected parameters in Dropout3d" << std::endl;
        }
        if (!buffers.empty()) {
            std::cerr << "Unexpected buffers in Dropout3d" << std::endl;
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}