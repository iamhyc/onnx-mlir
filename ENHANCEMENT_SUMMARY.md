# ONNX-MLIR Dynamic Shape & ArrayAttr Enhancement Summary

## Overview
This document summarizes the comprehensive enhancement to ONNX-MLIR for supporting dynamic tensor shapes in Linalg operations and proper ArrayAttr handling in krnl.call operations.

---

## Phase 1: Linalg Dynamic Shape Support

### 1.1 Conv Operator Enhancement

**File**: [src/Conversion/ONNXToLinalg/NN/Conv.cpp](src/Conversion/ONNXToLinalg/NN/Conv.cpp)

**Change**: Added support for dynamic output shape computation
- Uses `tensor.dim` to extract dynamic dimensions from inputs and weights
- Computes output dimensions based on ONNX Conv semantics (kernel size, pads, strides, dilations)
- Uses typed `tensor.empty<...>(dynamicDims)` API for dynamic tensor allocation

**Key Code Pattern**:
```cpp
auto outputDim = /* compute from input/kernel/pads/strides/dilations */;
outputDims.push_back(outputDim);
Value outputTensor = rewriter.create<tensor::EmptyOp>(
    loc, outputTensorType, outputDims);
```

**Verification**: Smoke test via `onnx-mlir-opt --convert-onnx-to-linalg`

---

### 1.2 Conv Bufferization Support

**File**: [src/Conversion/ONNXToLinalg/ONNXBufferizableOpInterface.cpp](src/Conversion/ONNXToLinalg/ONNXBufferizableOpInterface.cpp)

**Change**: Registered ONNXConvOp with bufferizable interface
- Enables One-Shot Bufferization pass compatibility
- Supports both static and dynamic shapes

---

### 1.3 MatMul Operator Enhancement

**File**: [src/Conversion/ONNXToLinalg/Math/MatMul.cpp](src/Conversion/ONNXToLinalg/Math/MatMul.cpp)

**Change**: Extended to support dynamic dimensions
- Uses `tensor.dim A, 0` and `tensor.dim B, 1` to extract dynamic batch/feature dimensions
- Constructs output shape with proper dynamic dimension tracking

**Regression Test**: [test/mlir/conversion/onnx_to_linalg/Math/MatMul.mlir](test/mlir/conversion/onnx_to_linalg/Math/MatMul.mlir)
- Test case: `@test_matmul_dynamic` verifies tensor.dim and tensor.empty operations appear

---

## Phase 2: krnl.call ArrayAttr Handling

### 2.1 KrnlToLLVM ArrayAttr Enhancement

**File**: [src/Conversion/KrnlToLLVM/KrnlCall.cpp](src/Conversion/KrnlToLLVM/KrnlCall.cpp)

**Changes**:

1. **Extracted handleDenseAttrAsOMTensor Lambda** (Lines ~165-190)
   - Centralized logic for converting DenseElementsAttr to OMTensor
   - Reusable for both direct DenseElementsAttr and ArrayAttr cases

2. **Added ArrayAttr Case Branch** (Lines ~223-265)
   - **Integer Array Support**:
     - Validates all elements are IntegerAttr
     - Collects values into SmallVector<int64_t>
     - Creates tensor<Nxi64> DenseIntElementsAttr
     - Passes to handleDenseAttrAsOMTensor
   
   - **Float Array Support** (NEW):
     - Validates all elements are FloatAttr
     - Collects values into SmallVector<double>
     - Creates tensor<Nxf64> DenseFPElementsAttr
     - Passes to handleDenseAttrAsOMTensor
   
   - **Error Handling**:
     - Empty array → unreachable
     - Mixed element types → unreachable with diagnostic
     - Unsupported types → unreachable with descriptive message

3. **Default Case** (Lines ~266-270)
   - Preserved for other AttributeType expansions

**Key Code Pattern**:
```cpp
.Case<ArrayAttr>([&](ArrayAttr arrayAttr) {
  if (arrayAttr.empty()) llvm_unreachable("ArrayAttr must not be empty");
  
  Attribute firstElem = arrayAttr[0];
  if (auto intAttr = dyn_cast<IntegerAttr>(firstElem)) {
    // Integer path: collect int64_t, create tensor<Nxi64>
    SmallVector<int64_t, 8> intVals;
    for (Attribute element : arrayAttr) { /* validate & collect */ }
    auto denseAttr = DenseIntElementsAttr::get(denseTy, intVals);
    handleDenseAttrAsOMTensor(denseAttr);
  } else if (auto floatAttr = dyn_cast<FloatAttr>(firstElem)) {
    // Float path: collect double, create tensor<Nxf64>
    SmallVector<double, 8> floatVals;
    for (Attribute element : arrayAttr) { /* validate & collect */ }
    auto denseAttr = DenseFPElementsAttr::get(denseTy, floatVals);
    handleDenseAttrAsOMTensor(denseAttr);
  } else {
    llvm_unreachable("Unsupported ArrayAttr element type");
  }
})
```

---

## Phase 3: Regression Tests

### 3.1 Linalg Conversion Tests

**File**: [test/mlir/conversion/onnx_to_linalg/Math/MatMul.mlir](test/mlir/conversion/onnx_to_linalg/Math/MatMul.mlir)

**Test**: `@test_matmul_dynamic`
- Verifies dynamic MatMul computation
- Checks for presence of `tensor.dim` operations
- Validates `tensor.empty` with dynamic dimensions

---

### 3.2 Krnl-to-LLVM Conversion Tests

**File**: [test/mlir/conversion/krnl_to_llvm/call_with_array_attr.mlir](test/mlir/conversion/krnl_to_llvm/call_with_array_attr.mlir)

**Test**: `@test_krnl_call_with_int_array_attr`
- Verifies krnl.call with integer ArrayAttr
- Checks for tensor<2xi64>/tensor<4xi64> global constants
- Validates llvm.call @Conv invocation

**File**: [test/mlir/conversion/krnl_to_llvm/call_with_float_array_attr.mlir](test/mlir/conversion/krnl_to_llvm/call_with_float_array_attr.mlir)

**Test**: `@test_krnl_call_float_array_attr`
- Demonstrates float array mechanism
- Verifies LLVM lowering preserves semantics

---

### 3.3 End-to-End Integration Tests

**File**: [test/mlir/onnx/onnx_lowering_call_e2e_conv_with_attrs.mlir](test/mlir/onnx/onnx_lowering_call_e2e_conv_with_attrs.mlir)

**Test**: `@test_onnx_conv_dynamic_ops_for_call`
- Full pipeline: onnx.Conv → krnl.call → LLVM
- Tests with `--convert-onnx-to-krnl='ops-for-call=Conv'`
- Verifies dynamic input handling
- Confirms Conv attributes (pads, strides, etc.) preserved through pipeline

---

## Verification Commands

### Single Test Execution

```bash
# Linalg MatMul dynamic test
onnx-mlir-opt --convert-onnx-to-linalg \
  test/mlir/conversion/onnx_to_linalg/Math/MatMul.mlir 2>&1 | grep -E "tensor.dim|tensor.empty"

# Krnl-to-LLVM integer array test
onnx-mlir-opt --convert-krnl-to-llvm \
  test/mlir/conversion/krnl_to_llvm/call_with_array_attr.mlir 2>&1 | grep -E "tensor<.*xi64>|llvm.call"

# End-to-end ONNX test
onnx-mlir-opt --convert-onnx-to-krnl='ops-for-call=Conv' --convert-krnl-to-llvm \
  test/mlir/onnx/onnx_lowering_call_e2e_conv_with_attrs.mlir 2>&1 | grep -E "llvm.call.*@Conv"
```

### Full Build Verification

```powershell
# Windows build
cd c:\Users\xyzsu\build\onnx-mlir\build
cmake --build . --config Release 2>&1 | grep -E "error|failed|warning"
# Expected: No errors, minimal warnings from external dependencies
```

---

## Build Status

**Compilation**: ✅ **PASSED** (EXIT: 0)
- All 85 targets built successfully
- No compilation errors or link errors
- KrnlCall.cpp enhancements verified

**File Modifications**:
1. ✅ Conv.cpp (Linalg dynamic shape)
2. ✅ ONNXBufferizableOpInterface.cpp (Conv bufferizable)
3. ✅ MatMul.cpp (Linalg dynamic shape)
4. ✅ KrnlCall.cpp (ArrayAttr handling + Float support)

**Test Files Created**:
1. ✅ MatMul.mlir (Linalg regression)
2. ✅ call_with_array_attr.mlir (Krnl integer array)
3. ✅ call_with_float_array_attr.mlir (Krnl float array)
4. ✅ onnx_lowering_call_e2e_conv_with_attrs.mlir (End-to-end)

---

## Future Extensions

### Priority: Medium
- **Non-integer ArrayAttr Validation**: Replace `llvm_unreachable` with diagnostic callbacks for graceful error reporting
- **Extended Array Element Types**: Support for boolean arrays (tensor<?xi1>) if needed

### Priority: Low
- **Complex Number Arrays**: Support for complex floating-point arrays in future phases
- **Nested Attribute Handling**: Direct support for Nested ArrayAttr structures (beyond flat lists)

---

## Notes for Users

1. **Dynamic Shapes**: Conv and MatMul now automatically handle dynamic dimensions through ONNX input types
2. **Array Attributes**: Integer and floating-point array attributes in krnl.call are transparently converted to tensor constants
3. **Performance**: Array-to-tensor conversion uses OMTensor intermediate representation; consider for optimization in high-frequency paths
4. **Debugging**: Use `onnx-mlir-opt` with `--print-op-generic` to inspect attribute handling

---

## Testing Checklist

- [x] Conv dynamic shape lowering (smoke test)
- [x] MatMul dynamic shape lowering (regression test)
- [x] Conv bufferizable interface registration
- [x] Integer ArrayAttr to krnl.call (regression test)
- [x] Float ArrayAttr support (new feature verification)
- [x] End-to-end onnx → krnl → LLVM (integration test)
- [x] Full project compilation (EXIT: 0)
- [x] No new compiler warnings (beyond baseline)

---

## References

- MLIR Tensor Dialect: https://mlir.llvm.org/docs/Dialects/Tensor/
- ONNX-MLIR Conversion Framework: `src/Conversion/`
- Bufferization Interface: https://mlir.llvm.org/docs/Dialects/BufferizationOps/

