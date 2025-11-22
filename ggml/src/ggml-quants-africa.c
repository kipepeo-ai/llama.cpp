// AfricaQuant integration for GGML
// This file provides wrapper functions to integrate AfricaQuant quantization
// with GGML's quantization system
//
// BUILD REQUIREMENTS:
// 1. This file must be compiled and linked with ggml-base
// 2. The kipepeo_quantization library must be linked (see CMakeLists.txt)
// 3. The AfricaQuant C API functions must be available at link time:
//    - kipepeo_quantize_1_28bit
//    - kipepeo_dequantize_1_28bit
//    - kipepeo_quantize_1_58bit
//    - kipepeo_dequantize_1_58bit

#include "ggml-quants.h"
#include "ggml-common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for AfricaQuant C API
// These should match the functions in kipepeo/quantization/africa_quant.h
// If the header is available, you can include it instead:
// #include "kipepeo/quantization/africa_quant.h"
extern bool kipepeo_quantize_1_28bit(
    const float* weights,
    size_t count,
    uint8_t* output,
    void* metadata,  // QuantizationMeta*
    uint32_t block_size
);

extern bool kipepeo_dequantize_1_28bit(
    const uint8_t* quantized,
    size_t count,
    float* output,
    const void* metadata,  // const QuantizationMeta*
    uint32_t block_size
);

extern bool kipepeo_quantize_1_58bit(
    const float* weights,
    size_t count,
    uint8_t* output,
    void* metadata,  // QuantizationMeta*
    uint32_t block_size
);

extern bool kipepeo_dequantize_1_58bit(
    const uint8_t* quantized,
    size_t count,
    float* output,
    const void* metadata,  // const QuantizationMeta*
    uint32_t block_size
);

// QuantizationMeta structure (must match kipepeo::quantization::QuantizationMeta)
typedef struct {
    float scale;
    float zero_point;
    uint32_t block_size;
    uint32_t codebook_size;
} QuantizationMeta;

// Quantize row for AfricaQuant 1.28-bit
void quantize_row_africa_1_28_ref(const float * GGML_RESTRICT x, block_africa_1_28 * GGML_RESTRICT y, int64_t k) {
    const int64_t nb = k / QK_AFRICA_1_28;
    
    for (int64_t i = 0; i < nb; i++) {
        const float * x_block = x + i * QK_AFRICA_1_28;
        block_africa_1_28 * y_block = y + i;
        
        // Prepare metadata structure
        QuantizationMeta meta;
        meta.block_size = QK_AFRICA_1_28;
        meta.codebook_size = 3; // Ternary quantization
        
        // Call AfricaQuant quantization
        bool success = kipepeo_quantize_1_28bit(
            x_block,
            QK_AFRICA_1_28,
            y_block->qs,
            &meta,
            QK_AFRICA_1_28
        );
        
        if (success) {
            // Copy metadata to block
            y_block->scale = meta.scale;
            y_block->zero_point = meta.zero_point;
        } else {
            // Fallback: zero out block on failure
            y_block->scale = 0.0f;
            y_block->zero_point = 0.0f;
            memset(y_block->qs, 0, sizeof(y_block->qs));
        }
    }
    
    // Handle remainder if k is not a multiple of QK_AFRICA_1_28
    const int64_t remainder = k % QK_AFRICA_1_28;
    if (remainder > 0) {
        const float * x_remainder = x + nb * QK_AFRICA_1_28;
        block_africa_1_28 * y_remainder = y + nb;
        
        // Pad remainder with zeros
        float x_padded[QK_AFRICA_1_28];
        memcpy(x_padded, x_remainder, remainder * sizeof(float));
        memset(x_padded + remainder, 0, (QK_AFRICA_1_28 - remainder) * sizeof(float));
        
        QuantizationMeta meta;
        meta.block_size = QK_AFRICA_1_28;
        meta.codebook_size = 3;
        
        bool success = kipepeo_quantize_1_28bit(
            x_padded,
            QK_AFRICA_1_28,
            y_remainder->qs,
            &meta,
            QK_AFRICA_1_28
        );
        
        if (success) {
            y_remainder->scale = meta.scale;
            y_remainder->zero_point = meta.zero_point;
        } else {
            y_remainder->scale = 0.0f;
            y_remainder->zero_point = 0.0f;
            memset(y_remainder->qs, 0, sizeof(y_remainder->qs));
        }
    }
}

// Dequantize row for AfricaQuant 1.28-bit
void dequantize_row_africa_1_28(const block_africa_1_28 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    const int64_t nb = k / QK_AFRICA_1_28;
    
    for (int64_t i = 0; i < nb; i++) {
        const block_africa_1_28 * x_block = x + i;
        float * y_block = y + i * QK_AFRICA_1_28;
        
        // Prepare metadata from block
        QuantizationMeta meta;
        meta.scale = x_block->scale;
        meta.zero_point = x_block->zero_point;
        meta.block_size = QK_AFRICA_1_28;
        meta.codebook_size = 3;
        
        // Call AfricaQuant dequantization
        bool success = kipepeo_dequantize_1_28bit(
            x_block->qs,
            QK_AFRICA_1_28,
            y_block,
            &meta,
            QK_AFRICA_1_28
        );
        
        if (!success) {
            // Fallback: zero out on failure
            memset(y_block, 0, QK_AFRICA_1_28 * sizeof(float));
        }
    }
    
    // Handle remainder
    const int64_t remainder = k % QK_AFRICA_1_28;
    if (remainder > 0) {
        const block_africa_1_28 * x_remainder = x + nb;
        float * y_remainder = y + nb * QK_AFRICA_1_28;
        
        QuantizationMeta meta;
        meta.scale = x_remainder->scale;
        meta.zero_point = x_remainder->zero_point;
        meta.block_size = QK_AFRICA_1_28;
        meta.codebook_size = 3;
        
        float temp[QK_AFRICA_1_28];
        bool success = kipepeo_dequantize_1_28bit(
            x_remainder->qs,
            QK_AFRICA_1_28,
            temp,
            &meta,
            QK_AFRICA_1_28
        );
        
        if (success) {
            memcpy(y_remainder, temp, remainder * sizeof(float));
        } else {
            memset(y_remainder, 0, remainder * sizeof(float));
        }
    }
}

// Quantize row for AfricaQuant 1.58-bit
void quantize_row_africa_1_58_ref(const float * GGML_RESTRICT x, block_africa_1_58 * GGML_RESTRICT y, int64_t k) {
    const int64_t nb = k / QK_AFRICA_1_58;
    
    for (int64_t i = 0; i < nb; i++) {
        const float * x_block = x + i * QK_AFRICA_1_58;
        block_africa_1_58 * y_block = y + i;
        
        QuantizationMeta meta;
        meta.block_size = QK_AFRICA_1_58;
        meta.codebook_size = 4; // Quaternary quantization
        
        bool success = kipepeo_quantize_1_58bit(
            x_block,
            QK_AFRICA_1_58,
            y_block->qs,
            &meta,
            QK_AFRICA_1_58
        );
        
        if (success) {
            y_block->scale = meta.scale;
            y_block->zero_point = meta.zero_point;
        } else {
            y_block->scale = 0.0f;
            y_block->zero_point = 0.0f;
            memset(y_block->qs, 0, sizeof(y_block->qs));
        }
    }
    
    const int64_t remainder = k % QK_AFRICA_1_58;
    if (remainder > 0) {
        const float * x_remainder = x + nb * QK_AFRICA_1_58;
        block_africa_1_58 * y_remainder = y + nb;
        
        float x_padded[QK_AFRICA_1_58];
        memcpy(x_padded, x_remainder, remainder * sizeof(float));
        memset(x_padded + remainder, 0, (QK_AFRICA_1_58 - remainder) * sizeof(float));
        
        QuantizationMeta meta;
        meta.block_size = QK_AFRICA_1_58;
        meta.codebook_size = 4;
        
        bool success = kipepeo_quantize_1_58bit(
            x_padded,
            QK_AFRICA_1_58,
            y_remainder->qs,
            &meta,
            QK_AFRICA_1_58
        );
        
        if (success) {
            y_remainder->scale = meta.scale;
            y_remainder->zero_point = meta.zero_point;
        } else {
            y_remainder->scale = 0.0f;
            y_remainder->zero_point = 0.0f;
            memset(y_remainder->qs, 0, sizeof(y_remainder->qs));
        }
    }
}

// Dequantize row for AfricaQuant 1.58-bit
void dequantize_row_africa_1_58(const block_africa_1_58 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    const int64_t nb = k / QK_AFRICA_1_58;
    
    for (int64_t i = 0; i < nb; i++) {
        const block_africa_1_58 * x_block = x + i;
        float * y_block = y + i * QK_AFRICA_1_58;
        
        QuantizationMeta meta;
        meta.scale = x_block->scale;
        meta.zero_point = x_block->zero_point;
        meta.block_size = QK_AFRICA_1_58;
        meta.codebook_size = 4;
        
        bool success = kipepeo_dequantize_1_58bit(
            x_block->qs,
            QK_AFRICA_1_58,
            y_block,
            &meta,
            QK_AFRICA_1_58
        );
        
        if (!success) {
            memset(y_block, 0, QK_AFRICA_1_58 * sizeof(float));
        }
    }
    
    const int64_t remainder = k % QK_AFRICA_1_58;
    if (remainder > 0) {
        const block_africa_1_58 * x_remainder = x + nb;
        float * y_remainder = y + nb * QK_AFRICA_1_58;
        
        QuantizationMeta meta;
        meta.scale = x_remainder->scale;
        meta.zero_point = x_remainder->zero_point;
        meta.block_size = QK_AFRICA_1_58;
        meta.codebook_size = 4;
        
        float temp[QK_AFRICA_1_58];
        bool success = kipepeo_dequantize_1_58bit(
            x_remainder->qs,
            QK_AFRICA_1_58,
            temp,
            &meta,
            QK_AFRICA_1_58
        );
        
        if (success) {
            memcpy(y_remainder, temp, remainder * sizeof(float));
        } else {
            memset(y_remainder, 0, remainder * sizeof(float));
        }
    }
}

#ifdef __cplusplus
}
#endif

