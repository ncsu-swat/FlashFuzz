#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Determine number of shapes to broadcast
        if (Size < 1) return 0;
        uint8_t num_shapes = (Data[offset++] % 5) + 1; // 1 to 5 shapes
        
        // Create vector to hold shapes
        std::vector<std::vector<int64_t>> shapes;
        
        // Parse shapes
        for (uint8_t i = 0; i < num_shapes && offset < Size; ++i) {
            // Get rank for this shape
            if (offset >= Size) break;
            uint8_t rank_byte = Data[offset++];
            uint8_t rank = fuzzer_utils::parseRank(rank_byte);
            
            // Parse shape dimensions
            auto shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
            shapes.push_back(shape);
        }
        
        // Call broadcast_shapes
        if (!shapes.empty()) {
            try {
                // Convert vector of shapes to vector of IntArrayRef
                std::vector<c10::IntArrayRef> shape_refs;
                for (const auto& shape : shapes) {
                    shape_refs.push_back(c10::IntArrayRef(shape));
                }
                
                // Call broadcast_shapes using c10::infer_size
                auto result = c10::infer_size(shape_refs[0], shape_refs.size() > 1 ? shape_refs[1] : shape_refs[0]);
                
                // For more than 2 shapes, iteratively broadcast
                for (size_t i = 2; i < shape_refs.size(); ++i) {
                    result = c10::infer_size(result, shape_refs[i]);
                }
                
                // Verify result by creating tensors with the shapes and checking if they broadcast
                if (!shapes.empty()) {
                    std::vector<torch::Tensor> tensors;
                    for (const auto& shape : shapes) {
                        tensors.push_back(torch::ones(shape));
                    }
                    
                    // Try to broadcast the tensors
                    if (tensors.size() >= 2) {
                        torch::Tensor broadcasted = tensors[0];
                        for (size_t i = 1; i < tensors.size(); ++i) {
                            broadcasted = broadcasted + tensors[i];
                        }
                        
                        // Check if the result shape matches what broadcast_shapes returned
                        auto tensor_shape = broadcasted.sizes().vec();
                        
                        if (tensor_shape != result) {
                            throw std::runtime_error("Broadcast shape mismatch");
                        }
                    }
                }
            } catch (const c10::Error& e) {
                // PyTorch-specific exceptions are expected for invalid inputs
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
