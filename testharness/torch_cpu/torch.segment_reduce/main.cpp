#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <algorithm>      // For std::max

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
        
        // Need at least a few bytes for basic operations
        if (Size < 8) {
            return 0;
        }
        
        // Parse reduction type from the first byte
        std::string reduction_type = "sum";
        uint8_t reduction_selector = Data[offset++];
        switch (reduction_selector % 4) {
            case 0: reduction_type = "sum"; break;
            case 1: reduction_type = "mean"; break;
            case 2: reduction_type = "max"; break;
            case 3: reduction_type = "min"; break;
        }
        
        // Parse axis and unsafe flag
        uint8_t axis_byte = Data[offset++];
        uint8_t flags_byte = Data[offset++];
        bool unsafe = (flags_byte & 0x01) != 0;
        bool use_offsets = (flags_byte & 0x02) != 0;
        bool use_initial = (flags_byte & 0x04) != 0;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is not empty and has at least 1 dimension
        if (input.numel() == 0 || input.dim() == 0) {
            input = torch::randn({4});
        }
        
        // Determine axis
        int64_t axis = axis_byte % input.dim();
        int64_t dim_size = input.size(axis);
        
        // Create lengths or offsets tensor that sums to dim_size
        torch::Tensor lengths = torch::Tensor();
        torch::Tensor offsets_tensor = torch::Tensor();
        
        if (dim_size > 0) {
            if (use_offsets) {
                // Create offsets: sorted indices including 0 and dim_size
                int64_t num_segments = 1;
                if (offset < Size) {
                    num_segments = std::max(1, static_cast<int>(Data[offset++] % std::min(dim_size, static_cast<int64_t>(8)))) + 1;
                }
                
                std::vector<int64_t> offsets_vec;
                offsets_vec.push_back(0);
                for (int64_t i = 1; i < num_segments; i++) {
                    int64_t off = (i * dim_size) / num_segments;
                    offsets_vec.push_back(off);
                }
                offsets_vec.push_back(dim_size);
                offsets_tensor = torch::tensor(offsets_vec, torch::kInt64);
            } else {
                // Create lengths that sum to dim_size
                int64_t num_segments = 1;
                if (offset < Size) {
                    num_segments = std::max(1, static_cast<int>(Data[offset++] % std::min(dim_size, static_cast<int64_t>(8)))) + 1;
                }
                
                std::vector<int64_t> lengths_vec;
                int64_t remaining = dim_size;
                for (int64_t i = 0; i < num_segments - 1; i++) {
                    int64_t len = remaining / (num_segments - i);
                    lengths_vec.push_back(len);
                    remaining -= len;
                }
                lengths_vec.push_back(remaining);
                lengths = torch::tensor(lengths_vec, torch::kInt64);
            }
        } else {
            lengths = torch::zeros({1}, torch::kInt64);
        }
        
        // Create initial value if requested
        c10::optional<torch::Scalar> initial = torch::nullopt;
        if (use_initial && offset < Size) {
            int8_t init_val = static_cast<int8_t>(Data[offset++]);
            initial = torch::Scalar(static_cast<double>(init_val));
        }
        
        // Try segment_reduce with lengths
        if (!use_offsets && lengths.defined()) {
            try {
                torch::Tensor result = torch::segment_reduce(
                    input, 
                    reduction_type, 
                    lengths,           // lengths
                    torch::nullopt,    // indices (not used with lengths)
                    offsets_tensor,    // offsets (nullopt when using lengths)
                    axis, 
                    unsafe,
                    initial
                );
            } catch (const c10::Error &e) {
                // Expected exceptions from invalid inputs
            }
        }
        
        // Try segment_reduce with offsets
        if (use_offsets && offsets_tensor.defined()) {
            try {
                torch::Tensor result = torch::segment_reduce(
                    input, 
                    reduction_type, 
                    torch::nullopt,    // lengths (nullopt when using offsets)
                    torch::nullopt,    // indices
                    offsets_tensor,    // offsets
                    axis, 
                    unsafe,
                    initial
                );
            } catch (const c10::Error &e) {
                // Expected exceptions from invalid inputs
            }
        }
        
        // Try with a contiguous input
        try {
            torch::Tensor contiguous_input = input.contiguous();
            torch::Tensor result = torch::segment_reduce(
                contiguous_input, 
                reduction_type, 
                use_offsets ? torch::Tensor() : lengths,
                torch::nullopt,
                use_offsets ? offsets_tensor : torch::Tensor(),
                axis, 
                unsafe,
                initial
            );
        } catch (const c10::Error &e) {
            // Expected exceptions
        }
        
        // Try with different dtype
        if (offset < Size && (Data[offset++] & 0x01)) {
            try {
                torch::Tensor float_input = input.to(torch::kFloat32);
                torch::Tensor result = torch::segment_reduce(
                    float_input, 
                    reduction_type, 
                    use_offsets ? torch::Tensor() : lengths,
                    torch::nullopt,
                    use_offsets ? offsets_tensor : torch::Tensor(),
                    axis, 
                    unsafe,
                    initial
                );
            } catch (const c10::Error &e) {
                // Expected exceptions
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}