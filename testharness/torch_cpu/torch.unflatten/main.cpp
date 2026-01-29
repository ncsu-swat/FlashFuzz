#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip scalar tensors - unflatten requires at least 1D
        if (input_tensor.dim() == 0) {
            return 0;
        }
        
        // Need at least 2 more bytes for dim and sizes
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Get dimension to unflatten - constrain to valid range
        int8_t raw_dim = static_cast<int8_t>(Data[offset++]);
        int64_t dim = raw_dim % input_tensor.dim();
        
        // Get number of dimensions to unflatten into
        uint8_t num_unflatten_dims = Data[offset++] % 4 + 1; // 1-4 dimensions
        
        // Get the size of the dimension we're unflattening
        int64_t dim_size = input_tensor.size(dim);
        
        // Skip if dimension size is 0
        if (dim_size == 0) {
            return 0;
        }
        
        // Parse unflatten sizes - they must multiply to dim_size
        std::vector<int64_t> unflatten_sizes;
        int64_t remaining = dim_size;
        
        for (uint8_t i = 0; i < num_unflatten_dims - 1 && offset < Size && remaining > 1; ++i) {
            // Choose a factor of remaining, or a small value that we'll adjust later
            int64_t size = static_cast<int64_t>(Data[offset++] % 4 + 1); // 1-4
            
            // Find the closest valid divisor
            while (size > 1 && remaining % size != 0) {
                size--;
            }
            
            if (size > 0 && remaining % size == 0) {
                unflatten_sizes.push_back(size);
                remaining /= size;
            }
        }
        
        // Add the remaining size as the last dimension
        unflatten_sizes.push_back(remaining);
        
        // If we ended up with just one size that equals dim_size, add another dimension of 1
        // to make unflatten meaningful (though unflatten with single dim is valid)
        if (unflatten_sizes.size() == 1 && offset < Size && Data[offset++] % 2 == 0) {
            unflatten_sizes.insert(unflatten_sizes.begin(), 1);
        }
        
        // Apply unflatten operation
        torch::Tensor output;
        
        // Try different variants of unflatten
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 2;
        }
        
        try {
            switch (variant) {
                case 0:
                    // Variant 1: unflatten with dimension and sizes vector
                    output = input_tensor.unflatten(dim, unflatten_sizes);
                    break;
                    
                case 1:
                    // Variant 2: unflatten with dimension and sizes as initializer list
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
        } catch (const c10::Error& e) {
            // Expected failures due to invalid dimension/size combinations
            return 0;
        }
        
        // Basic operations on the output tensor to ensure it's valid
        auto sizes = output.sizes();
        auto numel = output.numel();
        auto dtype = output.dtype();
        
        // Verify the unflatten was correct
        if (numel != input_tensor.numel()) {
            std::cerr << "Numel mismatch after unflatten!" << std::endl;
            return -1;
        }
        
        // Try some operations that might expose issues
        if (numel > 0 && output.is_floating_point()) {
            auto sum = output.sum();
            auto mean = output.mean();
        }
        
        // Verify we can flatten back
        try {
            auto flattened = output.flatten(dim, dim + static_cast<int64_t>(unflatten_sizes.size()) - 1);
            
            // Verify shapes match
            if (!flattened.sizes().equals(input_tensor.sizes())) {
                // Shape mismatch is unexpected
                std::cerr << "Shape mismatch after flatten roundtrip!" << std::endl;
            }
        } catch (const c10::Error& e) {
            // Flatten might fail in edge cases, that's ok
        }
        
        // Try contiguous and clone operations
        auto contiguous = output.contiguous();
        auto cloned = output.clone();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}