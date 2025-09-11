#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <sstream>        // For string operations
#include <algorithm>      // For string algorithms

// Simple textwrap implementation since torch::textwrap doesn't exist
namespace textwrap {
    std::vector<std::string> wrap(const std::string& text, int width) {
        std::vector<std::string> result;
        if (width <= 0) return result;
        
        std::istringstream iss(text);
        std::string word;
        std::string line;
        
        while (iss >> word) {
            if (line.empty()) {
                line = word;
            } else if (line.length() + 1 + word.length() <= static_cast<size_t>(width)) {
                line += " " + word;
            } else {
                result.push_back(line);
                line = word;
            }
        }
        if (!line.empty()) {
            result.push_back(line);
        }
        return result;
    }
    
    std::string fill(const std::string& text, int width) {
        auto lines = wrap(text, width);
        std::string result;
        for (size_t i = 0; i < lines.size(); ++i) {
            if (i > 0) result += "\n";
            result += lines[i];
        }
        return result;
    }
    
    std::string shorten(const std::string& text, int width) {
        if (static_cast<int>(text.length()) <= width) return text;
        if (width <= 3) return "...";
        return text.substr(0, width - 3) + "...";
    }
    
    std::string dedent(const std::string& text) {
        std::istringstream iss(text);
        std::string line;
        std::vector<std::string> lines;
        
        while (std::getline(iss, line)) {
            lines.push_back(line);
        }
        
        if (lines.empty()) return text;
        
        size_t min_indent = SIZE_MAX;
        for (const auto& l : lines) {
            if (l.empty()) continue;
            size_t indent = 0;
            while (indent < l.length() && (l[indent] == ' ' || l[indent] == '\t')) {
                indent++;
            }
            min_indent = std::min(min_indent, indent);
        }
        
        if (min_indent == SIZE_MAX) min_indent = 0;
        
        std::string result;
        for (size_t i = 0; i < lines.size(); ++i) {
            if (i > 0) result += "\n";
            if (lines[i].length() > min_indent) {
                result += lines[i].substr(min_indent);
            }
        }
        return result;
    }
    
    std::string indent(const std::string& text, const std::string& prefix) {
        std::istringstream iss(text);
        std::string line;
        std::string result;
        bool first = true;
        
        while (std::getline(iss, line)) {
            if (!first) result += "\n";
            result += prefix + line;
            first = false;
        }
        return result;
    }
}

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to use as input text
        torch::Tensor text_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a string tensor by converting to string
        std::string text;
        if (text_tensor.dim() == 0) {
            // Scalar tensor - convert to string
            text = std::to_string(text_tensor.item<float>());
        } else if (text_tensor.dim() == 1) {
            // 1D tensor - interpret as ASCII characters
            text.reserve(text_tensor.size(0));
            for (int64_t i = 0; i < text_tensor.size(0); i++) {
                char c = static_cast<char>(static_cast<int>(text_tensor[i].item<float>() * 128) % 128);
                if (c >= 32 && c < 127) { // Printable ASCII
                    text.push_back(c);
                } else {
                    text.push_back(' ');
                }
            }
        } else {
            // For higher dimensional tensors, flatten and use first few elements
            auto flat = text_tensor.flatten();
            int64_t len = std::min(static_cast<int64_t>(100), flat.size(0));
            text.reserve(len);
            for (int64_t i = 0; i < len; i++) {
                char c = static_cast<char>(static_cast<int>(flat[i].item<float>() * 128) % 128);
                if (c >= 32 && c < 127) { // Printable ASCII
                    text.push_back(c);
                } else {
                    text.push_back(' ');
                }
            }
        }
        
        // Parse width parameter from the remaining data
        int width = 70; // Default width
        if (offset + sizeof(int) <= Size) {
            std::memcpy(&width, Data + offset, sizeof(int));
            offset += sizeof(int);
            // Clamp width to reasonable range
            width = std::max(1, std::min(width, 1000));
        }
        
        // Parse expand_tabs parameter
        bool expand_tabs = true;
        if (offset < Size) {
            expand_tabs = Data[offset++] & 0x1;
        }
        
        // Parse replace_whitespace parameter
        bool replace_whitespace = true;
        if (offset < Size) {
            replace_whitespace = Data[offset++] & 0x1;
        }
        
        // Parse drop_whitespace parameter
        bool drop_whitespace = true;
        if (offset < Size) {
            drop_whitespace = Data[offset++] & 0x1;
        }
        
        // Parse initial_indent parameter
        std::string initial_indent = "";
        if (offset < Size) {
            int indent_len = Data[offset++] % 10; // Up to 10 spaces
            initial_indent = std::string(indent_len, ' ');
        }
        
        // Parse subsequent_indent parameter
        std::string subsequent_indent = "";
        if (offset < Size) {
            int indent_len = Data[offset++] % 10; // Up to 10 spaces
            subsequent_indent = std::string(indent_len, ' ');
        }
        
        // Parse fix_sentence_endings parameter
        bool fix_sentence_endings = false;
        if (offset < Size) {
            fix_sentence_endings = Data[offset++] & 0x1;
        }
        
        // Parse break_long_words parameter
        bool break_long_words = true;
        if (offset < Size) {
            break_long_words = Data[offset++] & 0x1;
        }
        
        // Parse break_on_hyphens parameter
        bool break_on_hyphens = true;
        if (offset < Size) {
            break_on_hyphens = Data[offset++] & 0x1;
        }
        
        // Parse tabsize parameter
        int tabsize = 8; // Default tabsize
        if (offset + sizeof(int) <= Size) {
            std::memcpy(&tabsize, Data + offset, sizeof(int));
            offset += sizeof(int);
            tabsize = std::max(1, std::min(tabsize, 32));
        }
        
        // Call textwrap.wrap
        auto wrapped = textwrap::wrap(text, width);
        
        // Call textwrap.fill
        auto filled = textwrap::fill(text, width);
        
        // Call textwrap.shorten
        int max_new_lines = 1;
        if (offset < Size) {
            max_new_lines = Data[offset++] % 5; // 0-4 new lines
        }
        
        auto shortened = textwrap::shorten(text, width);
        
        // Call textwrap.dedent
        auto dedented = textwrap::dedent(text);
        
        // Call textwrap.indent
        std::string prefix = "  "; // Default prefix
        if (offset < Size) {
            int prefix_len = Data[offset++] % 5; // 0-4 characters
            prefix = std::string(prefix_len, '>');
        }
        
        bool predicate = false;
        if (offset < Size) {
            predicate = Data[offset++] & 0x1;
        }
        
        auto indented = textwrap::indent(text, prefix);
        
        // Use the results to prevent optimization
        volatile size_t result_size = wrapped.size() + filled.size() + shortened.size() + 
                                     dedented.size() + indented.size();
        (void)result_size;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
