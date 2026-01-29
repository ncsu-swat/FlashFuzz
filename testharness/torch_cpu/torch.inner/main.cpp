#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create two input tensors for inner product
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            // Try the operation with just one tensor (against itself)
            // For inner product, the last dimensions must match
            try {
                torch::Tensor result = torch::inner(tensor1, tensor1);
            } catch (const std::exception&) {
                // Shape mismatch is expected, continue silently
            }
            return 0;
        }
        
        torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.inner operation
        // inner(input, other) -> Tensor
        // Computes the inner product of two tensors
        // The last dimension of both tensors must match
        try {
            torch::Tensor result = torch::inner(tensor1, tensor2);
        } catch (const std::exception&) {
            // Shape mismatches are expected with random tensors
        }
        
        // Try different variants if we have more data
        if (offset + 1 < Size) {
            uint8_t variant = Data[offset++];
            
            // Try with output tensor
            if (variant % 3 == 0) {
                try {
                    // Compute the expected output shape for inner product
                    // Result shape is tensor1.shape[:-1] + tensor2.shape[:-1]
                    std::vector<int64_t> output_shape;
                    if (tensor1.dim() > 0) {
                        for (int64_t i = 0; i < tensor1.dim() - 1; i++) {
                            output_shape.push_back(tensor1.size(i));
                        }
                    }
                    if (tensor2.dim() > 0) {
                        for (int64_t i = 0; i < tensor2.dim() - 1; i++) {
                            output_shape.push_back(tensor2.size(i));
                        }
                    }
                    
                    // Handle scalar case (empty shape means scalar)
                    if (output_shape.empty()) {
                        output_shape.push_back(1);
                    }
                    
                    // Create output tensor with appropriate shape and dtype
                    torch::Tensor output = torch::empty(output_shape, tensor1.options());
                    
                    // Call inner with output tensor
                    torch::inner_out(output, tensor1, tensor2);
                } catch (const std::exception&) {
                    // Shape calculation or operation may fail, continue silently
                }
            }
            
            // Try with different dtypes
            if (variant % 3 == 1) {
                try {
                    torch::Tensor t1_float = tensor1.to(torch::kFloat32);
                    torch::Tensor t2_float = tensor2.to(torch::kFloat32);
                    torch::Tensor result = torch::inner(t1_float, t2_float);
                } catch (const std::exception&) {
                    // Conversion or operation may fail
                }
            }
            
            // Try with contiguous tensors
            if (variant % 3 == 2) {
                try {
                    torch::Tensor t1_contig = tensor1.contiguous();
                    torch::Tensor t2_contig = tensor2.contiguous();
                    torch::Tensor result = torch::inner(t1_contig, t2_contig);
                } catch (const std::exception&) {
                    // Operation may fail due to shape mismatch
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}