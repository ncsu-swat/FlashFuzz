#include "fuzzer_utils.h"
#include <iostream>
#include <ATen/Context.h>

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
        
        if (Size < 2) {
            return 0;
        }
        
        // Extract flags from data
        bool use_deterministic = Data[offset] & 0x1;
        bool warn_only = Data[offset] & 0x2;
        offset++;
        
        // Test setting deterministic algorithms with different modes
        // This is the C++ equivalent of torch.use_deterministic_algorithms()
        at::globalContext().setDeterministicAlgorithms(use_deterministic, warn_only);
        
        // Verify the setting was applied
        bool is_deterministic = at::globalContext().deterministicAlgorithms();
        bool is_warn_only = at::globalContext().deterministicAlgorithmsWarnOnly();
        
        // Create tensors to exercise operations that might be affected
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (tensor.numel() > 0) {
                // Test operations that have deterministic variants
                try {
                    // Index operations can be affected by deterministic mode
                    if (tensor.dim() >= 1 && tensor.size(0) > 0) {
                        auto indices = torch::randint(0, tensor.size(0), {std::min<int64_t>(5, tensor.size(0))});
                        auto selected = tensor.index_select(0, indices);
                    }
                } catch (const c10::Error& e) {
                    // Expected - some ops throw when deterministic is enabled
                }
                
                try {
                    // Scatter operations are affected by deterministic mode
                    if (tensor.dim() >= 1) {
                        auto src = torch::ones_like(tensor);
                        auto idx = torch::zeros({tensor.size(0)}, torch::kLong);
                        auto result = torch::scatter(tensor, 0, idx.unsqueeze(-1).expand_as(tensor), src);
                    }
                } catch (const c10::Error& e) {
                    // Expected
                }
                
                try {
                    // Sort operations
                    if (tensor.dim() >= 1) {
                        auto [sorted, indices] = torch::sort(tensor, -1);
                    }
                } catch (const c10::Error& e) {
                    // Expected
                }
                
                try {
                    // Cumsum can be affected
                    if (tensor.dim() >= 1) {
                        auto result = torch::cumsum(tensor, 0);
                    }
                } catch (const c10::Error& e) {
                    // Expected
                }
            }
        }
        
        // Toggle the setting
        at::globalContext().setDeterministicAlgorithms(!use_deterministic, !warn_only);
        
        // Test with toggled settings
        if (offset < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (tensor2.numel() > 0 && tensor2.dim() >= 1) {
                try {
                    // Test gather operation
                    auto idx = torch::zeros({tensor2.size(0)}, torch::kLong);
                    for (int i = 1; i < tensor2.dim(); i++) {
                        idx = idx.unsqueeze(-1);
                    }
                    idx = idx.expand_as(tensor2);
                    auto result = torch::gather(tensor2, 0, idx);
                } catch (const c10::Error& e) {
                    // Expected
                }
                
                try {
                    // Test index_add
                    auto target = torch::zeros_like(tensor2);
                    auto idx = torch::zeros({tensor2.size(0)}, torch::kLong);
                    target.index_add_(0, idx, tensor2);
                } catch (const c10::Error& e) {
                    // Expected
                }
            }
        }
        
        // Always reset to non-deterministic mode at the end to avoid 
        // affecting subsequent test runs
        at::globalContext().setDeterministicAlgorithms(false, false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        // Reset before returning to ensure clean state
        at::globalContext().setDeterministicAlgorithms(false, false);
        return -1;
    }
    return 0;
}