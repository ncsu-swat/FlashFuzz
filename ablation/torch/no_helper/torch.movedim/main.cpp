#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for tensor creation and dimension parameters
        if (Size < 16) {
            return 0;
        }

        // Generate tensor dimensions (1-6 dimensions)
        auto num_dims = (Data[offset++] % 6) + 1;
        std::vector<int64_t> tensor_shape;
        
        for (int i = 0; i < num_dims; i++) {
            if (offset >= Size) return 0;
            // Keep dimensions reasonable (1-10) to avoid memory issues
            tensor_shape.push_back((Data[offset++] % 10) + 1);
        }

        // Create input tensor with random data
        auto input = torch::randn(tensor_shape);

        // Determine number of dimensions to move (1 to min(num_dims, 4))
        if (offset >= Size) return 0;
        auto num_moves = (Data[offset++] % std::min(num_dims, 4)) + 1;

        // Test single dimension move
        if (num_moves == 1) {
            if (offset + 1 >= Size) return 0;
            
            int source = static_cast<int>(Data[offset++]) % num_dims;
            int destination = static_cast<int>(Data[offset++]) % num_dims;
            
            // Test with positive indices
            auto result1 = torch::movedim(input, source, destination);
            
            // Test with negative indices
            int neg_source = source - num_dims;
            int neg_dest = destination - num_dims;
            auto result2 = torch::movedim(input, neg_source, neg_dest);
            
            // Results should be the same
            if (!torch::allclose(result1, result2)) {
                std::cout << "Mismatch between positive and negative indices" << std::endl;
            }
        }
        // Test multiple dimension moves
        else {
            std::vector<int> sources, destinations;
            std::set<int> used_sources, used_destinations;
            
            // Generate unique source indices
            for (int i = 0; i < num_moves && offset < Size; i++) {
                int source = static_cast<int>(Data[offset++]) % num_dims;
                // Ensure uniqueness
                while (used_sources.count(source) > 0) {
                    source = (source + 1) % num_dims;
                }
                sources.push_back(source);
                used_sources.insert(source);
            }
            
            // Generate unique destination indices
            for (int i = 0; i < num_moves && offset < Size; i++) {
                int dest = static_cast<int>(Data[offset++]) % num_dims;
                // Ensure uniqueness
                while (used_destinations.count(dest) > 0) {
                    dest = (dest + 1) % num_dims;
                }
                destinations.push_back(dest);
                used_destinations.insert(dest);
            }
            
            if (sources.size() == destinations.size() && !sources.empty()) {
                // Convert to tensors for the API call
                auto source_tensor = torch::tensor(sources, torch::kInt);
                auto dest_tensor = torch::tensor(destinations, torch::kInt);
                
                auto result = torch::movedim(input, source_tensor, dest_tensor);
                
                // Verify result has same number of elements
                if (result.numel() != input.numel()) {
                    std::cout << "Element count mismatch after movedim" << std::endl;
                }
                
                // Test with negative indices as well
                std::vector<int> neg_sources, neg_destinations;
                for (int s : sources) {
                    neg_sources.push_back(s - num_dims);
                }
                for (int d : destinations) {
                    neg_destinations.push_back(d - num_dims);
                }
                
                auto neg_source_tensor = torch::tensor(neg_sources, torch::kInt);
                auto neg_dest_tensor = torch::tensor(neg_destinations, torch::kInt);
                
                auto result_neg = torch::movedim(input, neg_source_tensor, neg_dest_tensor);
                
                // Results should be the same
                if (!torch::allclose(result, result_neg)) {
                    std::cout << "Mismatch between positive and negative indices in multi-dim move" << std::endl;
                }
            }
        }

        // Test edge cases
        if (offset < Size) {
            // Test moving dimension to same position (should be no-op)
            int same_dim = Data[offset++] % num_dims;
            auto no_op_result = torch::movedim(input, same_dim, same_dim);
            if (!torch::allclose(input, no_op_result)) {
                std::cout << "Moving dimension to same position should be no-op" << std::endl;
            }
        }

        // Test with different tensor types if we have more data
        if (offset < Size) {
            auto dtype_choice = Data[offset++] % 4;
            torch::Tensor typed_input;
            
            switch (dtype_choice) {
                case 0:
                    typed_input = input.to(torch::kFloat32);
                    break;
                case 1:
                    typed_input = input.to(torch::kFloat64);
                    break;
                case 2:
                    typed_input = input.to(torch::kInt32);
                    break;
                case 3:
                    typed_input = input.to(torch::kInt64);
                    break;
            }
            
            if (offset < Size) {
                int src = static_cast<int>(Data[offset++]) % num_dims;
                int dst = static_cast<int>(Data[offset++]) % num_dims;
                auto typed_result = torch::movedim(typed_input, src, dst);
                
                // Verify dtype is preserved
                if (typed_result.dtype() != typed_input.dtype()) {
                    std::cout << "Data type not preserved in movedim" << std::endl;
                }
            }
        }

        // Test with contiguous and non-contiguous tensors
        if (num_dims >= 2 && offset < Size) {
            auto transposed = input.transpose(0, 1);  // Make non-contiguous
            int src = static_cast<int>(Data[offset++]) % num_dims;
            int dst = static_cast<int>(Data[offset++]) % num_dims;
            
            auto result_contiguous = torch::movedim(input, src, dst);
            auto result_non_contiguous = torch::movedim(transposed, src, dst);
            
            // Both should succeed without error
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}