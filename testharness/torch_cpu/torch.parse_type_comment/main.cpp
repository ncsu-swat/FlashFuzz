#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/csrc/jit/frontend/parser.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for a meaningful test
        if (Size < 4) {
            return 0;
        }
        
        // Extract a string from the input data to use as a type comment
        std::string type_comment;
        size_t comment_length = std::min(Size - offset, static_cast<size_t>(Data[offset]) % 100);
        offset++;
        
        if (comment_length > 0 && offset + comment_length <= Size) {
            type_comment.assign(reinterpret_cast<const char*>(Data + offset), comment_length);
            offset += comment_length;
        } else {
            // Use some predefined type comments if we can't extract one
            const char* predefined_comments[] = {
                "# type: (int, str) -> bool",
                "# type: (List[int], Dict[str, Any]) -> None",
                "# type: (Tensor) -> Tensor",
                "# type: () -> None",
                "# type: invalid syntax here",
                "# not a type comment",
                "type: missing hash",
                "# type: (int,",  // incomplete
                "# type: -> bool", // missing param types
                "# type: (int) bool", // missing arrow
                "# type: (int) -> ", // missing return type
                "# type: (*args, **kwargs) -> Any"
            };
            
            type_comment = predefined_comments[Data[offset % 12]];
            offset++;
        }
        
        // Try to parse the type comment
        try {
            auto result = torch::jit::parseTypeComment(type_comment);
            
            // Use the result to prevent optimization
            if (result.has_value()) {
                auto parsed_type = result.value();
                // Do something with the parsed type to ensure it's used
                if (type_comment.find("Tensor") != std::string::npos) {
                    // Create a tensor if the type mentions Tensor
                    torch::Tensor t = fuzzer_utils::createTensor(Data, Size, offset);
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected for invalid type comments
        }
        
        // Try with different variations of the type comment
        if (offset < Size) {
            // Add some whitespace
            std::string modified_comment = "  " + type_comment + "  ";
            try {
                auto result = torch::jit::parseTypeComment(modified_comment);
            } catch (const c10::Error& e) {
                // Expected for invalid syntax
            }
            
            // Try with a malformed comment
            if (type_comment.length() > 3) {
                std::string malformed = type_comment;
                malformed[type_comment.length() / 2] = Data[offset % 256];
                try {
                    auto result = torch::jit::parseTypeComment(malformed);
                } catch (const c10::Error& e) {
                    // Expected for invalid syntax
                }
            }
            
            // Try with an empty string
            try {
                auto result = torch::jit::parseTypeComment("");
            } catch (const c10::Error& e) {
                // Expected for invalid syntax
            }
            
            // Try with a very long comment if we have enough data
            if (Size > offset + 100) {
                std::string long_comment = "# type: (";
                for (size_t i = 0; i < 50 && offset + i < Size; i++) {
                    long_comment += "int, ";
                }
                long_comment += "int) -> bool";
                try {
                    auto result = torch::jit::parseTypeComment(long_comment);
                } catch (const c10::Error& e) {
                    // Expected for invalid syntax
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