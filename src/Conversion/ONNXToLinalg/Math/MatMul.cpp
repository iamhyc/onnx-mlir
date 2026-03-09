//===------------- MatMul.cpp - Lowering MatMul Op to Linalg -------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX MatMul Operator to Linalg dialect.
//
// Current support: 2D x 2D matrix multiplication only
// TODO: Add support for:
//   - 1D x 2D (vector-matrix multiplication)
//   - 2D x 1D (matrix-vector multiplication)
//   - ND x ND (batch matmul with broadcasting)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToLinalg/ONNXToLinalgCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXMatMulOpLoweringToLinalg : public OpRewritePattern<ONNXMatMulOp> {
  ONNXMatMulOpLoweringToLinalg(
      MLIRContext *ctx, const std::string &linalgOps, bool useLinalgPath)
      : OpRewritePattern<ONNXMatMulOp>(ctx), linalgOps(linalgOps),
        useLinalgPath(useLinalgPath) {}

  LogicalResult matchAndRewrite(
      ONNXMatMulOp matMulOp, PatternRewriter &rewriter) const final {
    // Check if this operation should be converted to Linalg based on
    // --linalg-ops option
    Operation *op = matMulOp.getOperation();
    if (!shouldConvertToLinalg(op, linalgOps, useLinalgPath)) {
      return rewriter.notifyMatchFailure(
          matMulOp, "operation not selected for Linalg conversion");
    }

    ONNXMatMulOpAdaptor adaptor(matMulOp);

    Location loc = matMulOp.getLoc();
    Value A = adaptor.getA();
    Value B = adaptor.getB();

    // Get input types
    auto aType = dyn_cast<RankedTensorType>(A.getType());
    auto bType = dyn_cast<RankedTensorType>(B.getType());
    if (!aType || !bType)
      return rewriter.notifyMatchFailure(
          matMulOp, "expected ranked tensor types");

    // [GUARD] Only support 2D x 2D matmul for now
    // Reject 1D, 3D, and higher dimensional inputs (batch matmul)
    if (aType.getRank() != 2 || bType.getRank() != 2) {
      return rewriter.notifyMatchFailure(matMulOp,
          "only 2D x 2D MatMul is currently supported in Linalg lowering");
    }

    // Get output type
    Type outputType = matMulOp.getResult().getType();
    auto outputTensorType = dyn_cast<RankedTensorType>(outputType);
    if (!outputTensorType)
      return rewriter.notifyMatchFailure(
          matMulOp, "expected ranked output tensor type");

    ArrayRef<int64_t> outputShape = outputTensorType.getShape();
    SmallVector<Value, 2> dynamicOutputDims;

    if (ShapedType::isDynamic(outputShape[0]))
      dynamicOutputDims.emplace_back(tensor::DimOp::create(rewriter, loc, A, 0));
    if (ShapedType::isDynamic(outputShape[1]))
      dynamicOutputDims.emplace_back(tensor::DimOp::create(rewriter, loc, B, 1));

    // Create output tensor with tensor.empty
    Value emptyTensor =
      tensor::EmptyOp::create(rewriter, loc, outputTensorType, dynamicOutputDims);

    // Create zero constant for initialization
    Value zero = arith::ConstantOp::create(rewriter, loc,
        outputTensorType.getElementType(),
        rewriter.getZeroAttr(outputTensorType.getElementType()));

    // Fill the output tensor with zeros
    Value filledTensor = linalg::FillOp::create(
        rewriter, loc, ValueRange{zero}, ValueRange{emptyTensor})
                             .getResult(0);

    // Create linalg.matmul operation
    Value matmulResult = linalg::MatmulOp::create(
        rewriter, loc, ValueRange{A, B}, ValueRange{filledTensor})
                             .getResult(0);

    rewriter.replaceOp(matMulOp, matmulResult);
    return success();
  }

private:
  std::string linalgOps;
  bool useLinalgPath;
};

void populateLoweringONNXMatMulOpToLinalgPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx,
    const std::string &linalgOps, bool useLinalgPath) {
  patterns.add<ONNXMatMulOpLoweringToLinalg>(ctx, linalgOps, useLinalgPath);
}

} // namespace onnx_mlir
