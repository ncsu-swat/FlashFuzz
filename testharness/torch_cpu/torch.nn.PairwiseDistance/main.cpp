#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        
        // Create first input tensor
        torch::Tensor x1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor with same shape as x1 for pairwise distance
        torch::Tensor x2;
        if (offset < Size) {
            x2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            x2 = torch::randn_like(x1);
        }
        
        // Extract parameters for PairwiseDistance from remaining data
        double p = 2.0;
        bool keepdim = false;
        double eps = 1e-6;
        
        if (offset < Size) {
            uint8_t p_selector = Data[offset++];
            if (p_selector % 4 == 0) {
                p = 1.0;
            } else if (p_selector % 4 == 1) {
                p = 2.0;
            } else if (p_selector % 4 == 2) {
                p = 3.0;
            } else {
                p = 0.5 + (p_selector % 10);
            }
        }
        
        if (offset < Size) {
            keepdim = (Data[offset++] % 2 == 0);
        }
        
        if (offset < Size) {
            uint8_t eps_selector = Data[offset++];
            eps = 1e-8 * (1 + eps_selector % 100);
        }
        
        // Ensure tensors are floating point for distance computation
        if (!x1.is_floating_point()) {
            x1 = x1.to(torch::kFloat32);
        }
        if (!x2.is_floating_point()) {
            x2 = x2.to(torch::kFloat32);
        }
        
        // PairwiseDistance requires tensors with the same shape
        // Make x2 match x1's shape
        if (x1.sizes() != x2.sizes()) {
            try {
                // Try to expand/broadcast x2 to match x1
                x2 = x2.expand_as(x1).clone();
            } catch (...) {
                // If expansion fails, create new tensor with same shape
                x2 = torch::randn_like(x1);
            }
        }
        
        // Ensure tensors are at least 1D
        if (x1.dim() == 0) {
            x1 = x1.unsqueeze(0);
            x2 = x2.unsqueeze(0);
        }
        
        // Create PairwiseDistance module
        auto options = torch::nn::PairwiseDistanceOptions().p(p).eps(eps).keepdim(keepdim);
        torch::nn::PairwiseDistance pairwise_distance(options);
        
        // Apply pairwise distance using module
        torch::Tensor output;
        try {
            output = pairwise_distance(x1, x2);
        } catch (...) {
            // Shape mismatch or other expected errors
            return 0;
        }
        
        // Also test the functional interface
        if (offset < Size && Data[offset - 1] % 2 == 0) {
            try {
                torch::Tensor output_func = torch::pairwise_distance(x1, x2, p, eps, keepdim);
                (void)output_func;
            } catch (...) {
                // Silently handle expected errors
            }
        }
        
        // Test with different p values using functional API
        if (offset < Size) {
            uint8_t test_selector = Data[offset - 1];
            try {
                if (test_selector % 5 == 0) {
                    // L1 distance
                    auto out1 = torch::pairwise_distance(x1, x2, 1.0, eps, keepdim);
                    (void)out1;
                } else if (test_selector % 5 == 1) {
                    // L2 distance
                    auto out2 = torch::pairwise_distance(x1, x2, 2.0, eps, keepdim);
                    (void)out2;
                } else if (test_selector % 5 == 2) {
                    // Infinity norm (max)
                    auto out_inf = torch::pairwise_distance(x1, x2, std::numeric_limits<double>::infinity(), eps, keepdim);
                    (void)out_inf;
                }
            } catch (...) {
                // Silently handle expected errors
            }
        }
        
        // Test pdist for 2D tensors (computes pairwise distances between all rows)
        if (x1.dim() == 2 && x1.size(0) > 1) {
            try {
                auto pdist_output = torch::pdist(x1, p);
                (void)pdist_output;
            } catch (...) {
                // Silently handle expected errors
            }
        }
        
        // Access output to ensure computation is performed
        if (output.defined() && output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}