#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for unflatten operation
        // Need at least 2 more bytes for dim and sizes
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Get dimension to unflatten
        int64_t dim = static_cast<int64_t>(Data[offset++]);
        
        // Allow negative dimensions (PyTorch handles them by wrapping)
        if (dim > 127) {
            dim = dim - 256; // Convert to negative value if high bit is set
        }
        
        // Get number of dimensions to unflatten into
        uint8_t num_unflatten_dims = Data[offset++] % 4 + 1; // 1-4 dimensions
        
        // Parse unflatten sizes
        std::vector<int64_t> unflatten_sizes;
        for (uint8_t i = 0; i < num_unflatten_dims && offset < Size; ++i) {
            int64_t size = static_cast<int64_t>(Data[offset++] % 8 + 1); // 1-8 size
            unflatten_sizes.push_back(size);
        }
        
        // If we don't have enough dimensions, add some defaults
        while (unflatten_sizes.size() < num_unflatten_dims) {
            unflatten_sizes.push_back(1);
        }
        
        // Apply unflatten operation
        torch::Tensor output;
        
        // Try different variants of unflatten
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 3;
            
            switch (variant) {
                case 0:
                    // Variant 1: unflatten with dimension and sizes vector
                    output = input_tensor.unflatten(dim, unflatten_sizes);
                    break;
                    
                case 1:
                    // Variant 2: unflatten with named dimension (if tensor has names)
                    if (unflatten_sizes.size() >= 2) {
                        // Create some dimension names
                        std::vector<torch::Dimname> dimnames;
                        for (size_t i = 0; i < input_tensor.dim(); ++i) {
                            dimnames.push_back(torch::Dimname::wildcard());
                        }
                        
                        // Apply names to the tensor
                        auto named_tensor = input_tensor.refine_names(dimnames);
                        
                        // Create names for the unflattened dimensions
                        std::vector<torch::Dimname> unflatten_names;
                        for (size_t i = 0; i < unflatten_sizes.size(); ++i) {
                            unflatten_names.push_back(torch::Dimname::wildcard());
                        }
                        
                        // Use the first dimname for unflattening with names
                        output = named_tensor.unflatten(dimnames[0], unflatten_sizes, unflatten_names);
                    } else {
                        output = input_tensor.unflatten(dim, unflatten_sizes);
                    }
                    break;
                    
                case 2:
                    // Variant 3: unflatten with dimension and sizes as initializer list
                    if (unflatten_sizes.size() == 1) {
                        output = input_tensor.unflatten(dim, {unflatten_sizes[0]});
                    } else if (unflatten_sizes.size() == 2) {
                        output = input_tensor.unflatten(dim, {unflatten_sizes[0], unflatten_sizes[1]});
                    } else if (unflatten_sizes.size() == 3) {
                        output = input_tensor.unflatten(dim, {unflatten_sizes[0], unflatten_sizes[1], unflatten_sizes[2]});
                    } else {
                        output = input_tensor.unflatten(dim, {unflatten_sizes[0], unflatten_sizes[1], unflatten_sizes[2], unflatten_sizes[3]});
                    }
                    break;
            }
        } else {
            // Default variant if we don't have enough data
            output = input_tensor.unflatten(dim, unflatten_sizes);
        }
        
        // Basic operations on the output tensor to ensure it's valid
        auto sizes = output.sizes();
        auto numel = output.numel();
        auto dtype = output.dtype();
        
        // Try some operations that might expose issues
        if (numel > 0) {
            auto sum = output.sum();
            auto mean = output.mean();
        }
        
        // Try reshaping back to original
        auto reshaped = output.reshape_as(input_tensor);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}