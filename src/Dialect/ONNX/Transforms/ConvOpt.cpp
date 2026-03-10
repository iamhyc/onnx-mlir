/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- ConvOpt.cpp - ONNX high level Convolution Optimizations ---------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of optimizations to optimize the execution of
// convolutions on CPUs.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXLayoutHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/Mlir/VectorMachineSupport.hpp"
#include "src/Dialect/ONNX/Transforms/ConvOpt.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"

// Enables a minimum of printing.
#define DEBUG 0

using namespace mlir;

namespace onnx_mlir {

// Determine if we can transform a conv 1x1 with group=1, kernel size =1x...x1,
// stride=dilation=1, pad=0.
bool ExpressONNXConvOpAsMatmul(ONNXConvOp convOp, bool verbose = 0) {
  // Get type, shape, and rank info for X and W inputs.
  Value X = convOp.getX();
  Value W = convOp.getW();
  Value B = convOp.getB();
  bool hasBias = !isNoneValue(B);
  if (!hasShapeAndRank(X) || !hasShapeAndRank(W))
    return false;
  if (hasBias && !hasShapeAndRank(B))
    return false;
  ShapedType xType = mlir::cast<ShapedType>(X.getType());
  ShapedType wType = mlir::cast<ShapedType>(W.getType());
  auto xShape = xType.getShape();
  auto wShape = wType.getShape();
  int64_t rank = xShape.size();
  assert(rank == (int64_t)wShape.size() && "X and W should have same rank");
  assert(rank > 2 && "X and W should have to spatial dims");
  // Compute spatial rank: all but N & Cin in X, Cout & Cin in W.
  int spatialRank = rank - 2;
  int spatialIndex = 2;
  // Eliminate conv ops with groups > 1.
  int G = convOp.getGroup();
  if (G != 1)
    return false;
  // Eliminating conv with spacial dims of the kernel that are not 1.
  for (int i = spatialIndex; i < rank; ++i)
    if (wShape[i] != 1)
      return false;
  // Eliminate conv op with dilations>1.
  auto dilations = convOp.getDilations();
  if (dilations.has_value()) {
    for (int i = 0; i < spatialRank; ++i)
      if (ArrayAttrIntVal(dilations, i) != 1)
        return false;
  }
  // ELiminate conv ops with strides>1.
  auto strides = convOp.getStrides();
  if (strides.has_value()) {
    for (int i = 0; i < spatialRank; ++i)
      if (ArrayAttrIntVal(strides, i) != 1)
        return false;
  }
  // Eliminate conv ops with any padding.
  auto autoPad = convOp.getAutoPad();
  if (autoPad == "NOTSET") {
    // Explicitly given padding, check that it is all zero. Don't have to
    // worry about the other cases (SAME_UPPER/LOWER, VALID), as with 1x1
    // kernel of stride/dilation of 1, there is never any padding for the
    // (deprecated) automatic padding options.
    auto pads = convOp.getPads();
    if (pads.has_value()) {
      for (int i = 0; i < 2 * spatialRank; ++i) // 2x for before/after.
        if (ArrayAttrIntVal(pads, i) != 0)
          return false;
    }
  }
  if (verbose)
    printf("optimize conv 1x1 with matmul: N %d, group %d, Ci %d, Co %d, H %d, "
           "W %d, rank %d, with%s bias\n",
        (int)xShape[0], (int)G, (int)xShape[1], (int)wShape[0], (int)xShape[2],
        (int)xShape[3], (int)rank, hasBias ? "" : "out");
  return true;
}

} // namespace onnx_mlir

namespace {

int64_t getSimdLayoutFactorForType(Type elementType) {
  if (!onnx_mlir::VectorMachineSupport::hasSimd())
    return 4;
  int64_t archVL =
      onnx_mlir::VectorMachineSupport::getArchVectorLength(elementType);
  // Keep factors to known/stable encodings for now.
  if (archVL >= 8)
    return 8;
  return 4;
}

bool isLikelyProfitableForSimdLayout(
    ONNXConvOp convOp, Type elementType, int64_t layoutFactor) {
  auto xType = mlir::dyn_cast<RankedTensorType>(convOp.getX().getType());
  auto wType = mlir::dyn_cast<RankedTensorType>(convOp.getW().getType());
  auto yType = mlir::dyn_cast<RankedTensorType>(convOp.getY().getType());
  if (!xType || !wType || !yType)
    return false;

  // Focus on convs with enough channel width to amortize layout transforms.
  if (!xType.hasRank() || !wType.hasRank() || xType.getRank() < 2 ||
      wType.getRank() < 2)
    return false;
  int64_t cin = xType.getShape()[1];
  int64_t cout = wType.getShape()[0];
  if (cin == ShapedType::kDynamic || cout == ShapedType::kDynamic)
    return false;
  if (cin < 2 * layoutFactor || cout < 2 * layoutFactor)
    return false;

  return true;
}

bool hasConvLayoutEncoding(Value v) {
  return onnx_mlir::hasConvONNXTensorDataLayout(v.getType());
}

ONNXTensorEncodingAttr getConvEncodingOrNull(Value v) {
  return onnx_mlir::getONNXTensorEncoding(v.getType());
}

bool canKeepOutputInSimdLayout(ONNXConvOp convOp) {
  for (OpOperand &use : convOp.getY().getUses()) {
    if (!mlir::isa<ONNXConvOp>(use.getOwner()))
      return false;
    // Keep layout only when feeding the X input of downstream conv.
    if (use.getOperandNumber() != 0)
      return false;
  }
  return true;
}

bool shouldAttemptSimdLayoutRewrite(ONNXConvOp convOp) {
  if (!onnx_mlir::isRankedShapedType(convOp.getY().getType()))
    return false;

  auto yType = mlir::dyn_cast<RankedTensorType>(convOp.getY().getType());
  if (!yType)
    return false;

  bool xHasLayout = hasConvLayoutEncoding(convOp.getX());
  bool wHasLayout = hasConvLayoutEncoding(convOp.getW());
  int64_t layoutFactor = 0;
  if (xHasLayout) {
    auto xEnc = getConvEncodingOrNull(convOp.getX());
    if (!xEnc ||
        xEnc.getDataLayout() != ONNXTensorEncodingAttr::DataLayout::NCHWxC)
      return false;
    layoutFactor = xEnc.getXFactor();
  } else if (wHasLayout) {
    auto wEnc = getConvEncodingOrNull(convOp.getW());
    if (!wEnc ||
        wEnc.getDataLayout() != ONNXTensorEncodingAttr::DataLayout::KCNMxCyK)
      return false;
    layoutFactor = wEnc.getXFactor();
  } else {
    layoutFactor = getSimdLayoutFactorForType(yType.getElementType());
  }

  if (!isLikelyProfitableForSimdLayout(
          convOp, yType.getElementType(), layoutFactor))
    return false;
  if (!canKeepOutputInSimdLayout(convOp) && !xHasLayout && !wHasLayout)
    return false;
  return true;
}

struct ConvToSimdLayoutPattern : public ConversionPattern {
  ConvToSimdLayoutPattern(MLIRContext *context)
      : ConversionPattern(ONNXConvOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXConvOp convOp = llvm::cast<ONNXConvOp>(op);
    Value x = convOp.getX();
    Value w = convOp.getW();
    Value b = convOp.getB();

    if (!shouldAttemptSimdLayoutRewrite(convOp))
      return failure();

    auto yType = mlir::dyn_cast<RankedTensorType>(convOp.getY().getType());
    if (!yType)
      return failure();

    bool xHasLayout = hasConvLayoutEncoding(x);
    bool wHasLayout = hasConvLayoutEncoding(w);
    int64_t layoutFactor = 0;

    if (xHasLayout) {
      auto xEnc = getConvEncodingOrNull(x);
      if (!xEnc ||
          xEnc.getDataLayout() != ONNXTensorEncodingAttr::DataLayout::NCHWxC)
        return failure();
      layoutFactor = xEnc.getXFactor();
    } else if (wHasLayout) {
      auto wEnc = getConvEncodingOrNull(w);
      if (!wEnc ||
          wEnc.getDataLayout() != ONNXTensorEncodingAttr::DataLayout::KCNMxCyK)
        return failure();
      layoutFactor = wEnc.getXFactor();
    } else {
      layoutFactor = getSimdLayoutFactorForType(yType.getElementType());
    }

    bool keepOutputInSimd = canKeepOutputInSimdLayout(convOp);

    Attribute xLayoutAttr = ONNXTensorEncodingAttr::get(rewriter.getContext(),
        ONNXTensorEncodingAttr::DataLayout::NCHWxC, layoutFactor, 0);
    Attribute wLayoutAttr = ONNXTensorEncodingAttr::get(rewriter.getContext(),
        ONNXTensorEncodingAttr::DataLayout::KCNMxCyK, layoutFactor,
        layoutFactor);

    Location loc = convOp.getLoc();
    Value xOpt = x;
    if (!xHasLayout)
      xOpt =
          rewriter.create<ONNXLayoutTransformOp>(loc, x, xLayoutAttr).getOutput();

    Value wOpt = w;
    if (!wHasLayout)
      wOpt =
          rewriter.create<ONNXLayoutTransformOp>(loc, w, wLayoutAttr).getOutput();

    Type convResType = onnx_mlir::convertTensorTypeToTensorTypeWithEncoding(
        convOp.getY().getType(), xLayoutAttr);
    OperationState convState(loc, ONNXConvOp::getOperationName());
    convState.addTypes(convResType);
    convState.addOperands({xOpt, wOpt, b});
    convState.addAttributes(convOp->getAttrs());
    Operation *newConv = rewriter.create(convState);

    if (keepOutputInSimd) {
      rewriter.replaceOp(convOp, newConv->getResult(0));
    } else {
      Value res = rewriter
                      .create<ONNXLayoutTransformOp>(
                          loc, newConv->getResult(0), Attribute())
                      .getOutput();
      rewriter.replaceOp(convOp, res);
    }
    return success();
  }
};

/*
   Pattern: when we have a convolution with filter of 1x1, stride 1, dilation of
   1, group of 1, and no padding; then we can perform the following
   transformation.

   from:
     res = CONV(X=<NxCIxHxW>, W=<COxCIx1x1>)
   to:
     XX = reshape(X, <N, CO, H*W>) // flatten the last 2 dims.
     WW = squeeze(W) // get rid of the last 2 1s in the dims.
     MM = matmul(WW, XX) //  <CO, CI> * <N, CI, H*W> = <N, CO, H*W>
     if (has bias) {
        BB = unsqueeze(B, {0,2}) // <CO> -> <1, CO, 1>
        MM = add(MM, BB)
     }
     res = reshape(MM, <N, CO, H, W>)

   Note: since there is no pad, dilation, stride, the output spacial dims (H, W)
   are the same on inputs and outputs.
*/

struct Conv1x1ToMatmulPattern : public ConversionPattern {
  Conv1x1ToMatmulPattern(MLIRContext *context)
      : ConversionPattern(ONNXConvOp::getOperationName(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Get basic op info.
    ONNXConvOp convOp = ::llvm::dyn_cast<ONNXConvOp>(op);
    Location loc = convOp.getLoc();
    // All conditions should be satisfied, test to be sure.
    if (!onnx_mlir::ExpressONNXConvOpAsMatmul(convOp, DEBUG))
      return failure();
    if (DEBUG)
      fprintf(
          stderr, "ConvOps match&rewrite: go for the actual conv 1x1 opt.\n");
    // All conditions satisfied, get info.
    Value X = convOp.getX();
    Value W = convOp.getW();
    Value B = convOp.getB();
    bool hasBias = !onnx_mlir::isNoneValue(B);
    ShapedType xType = mlir::cast<ShapedType>(X.getType());
    ShapedType wType = mlir::cast<ShapedType>(W.getType());
    Type elementType = xType.getElementType();
    auto xShape = xType.getShape();
    auto wShape = wType.getShape();
    int64_t rank = xShape.size();
    int64_t spatialRank = rank - 2;
    // Get dimensions.
    int64_t batchSize = xShape[0];
    int64_t Cout = wShape[0];
    // Start transforming.
    onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(
        rewriter, loc);
    // Reshape [N, CI, H, W,...] to [N, CI, H*W*...] by collapsing all spatial
    // dims.
    Value XX =
        create.onnx.reshapeToNDim(X, 3, /*collapseMostSignificant*/ false);
    // Squeeze <Cout, Cin, 1, 1, ...> can be implemented by a reshape to <Cout,
    // *>, collapsing all spatial dims.
    Value WW =
        create.onnx.reshapeToNDim(W, 2, /*collapseMostSignificant*/ false);
    // Perform the matrix multiplication on WW * XX. Leave last dim runtime so
    // that its actual H*W size can be generated during shape inference.
    RankedTensorType MMOutputType = RankedTensorType::get(
        {batchSize, Cout, ShapedType::kDynamic}, elementType);
    Value MM = create.onnx.matmul(MMOutputType, WW, XX, /*gemm*/ false);
    if (hasBias) {
      // Reshape BB from <CO> to <1, CO, 1> for broadcast.
      Value axes = create.onnx.constantInt64({0, 2});
      Type bbType = RankedTensorType::get({1, Cout, 1}, elementType);
      Value BB = create.onnx.unsqueeze(bbType, B, axes);
      MM = create.onnx.add(MM, BB);
    }
    // Get type for shapes
    Type shapeType = RankedTensorType::get({rank}, rewriter.getI64Type());
    Type batchCoutShapeType = RankedTensorType::get({1}, rewriter.getI64Type());
    Type spatialShapeType =
        RankedTensorType::get({spatialRank}, rewriter.getI64Type());
    // Get shape value from X, W.
    Value xShapeVals = create.onnx.shape(shapeType, X);
    Value wShapeVals = create.onnx.shape(shapeType, W);
    Value batchShapeVal =
        create.onnx.slice(batchCoutShapeType, xShapeVals, 0, 1);
    Value CoutShapeVal =
        create.onnx.slice(batchCoutShapeType, wShapeVals, 0, 1);
    Value spatialShapeVal =
        create.onnx.slice(spatialShapeType, xShapeVals, 2, rank);
    // Output shape values: batch, Cout, spatial shape values
    Value outputShapeVals = create.onnx.concat(
        shapeType, {batchShapeVal, CoutShapeVal, spatialShapeVal}, 0);
    // Output type is the same as input, except for Cin becomes Cout.
    llvm::SmallVector<int64_t, 4> outputDims;
    for (int i = 0; i < rank; ++i)
      outputDims.emplace_back(xShape[i]);
    outputDims[1] = Cout;
    Value res =
        create.onnx.reshape(convOp.getY().getType(), MM, outputShapeVals);
    // Replace op and declare success.
    rewriter.replaceOp(convOp, res);
    return success();
  }
};

} // namespace

void onnx_mlir::getConvOptONNXToONNXPatterns(
    bool enableSimdDataLayoutOpt, RewritePatternSet &patterns) {
  // TODO: if enable simd layout opt, we still need to determine how 1x1 and
  // simd layout interact. Right now, only enable the one or the other. Will
  // need to refine this later.
  if (enableSimdDataLayoutOpt)
    patterns.insert<ConvToSimdLayoutPattern>(patterns.getContext());
  else
    patterns.insert<Conv1x1ToMatmulPattern>(patterns.getContext());
}

namespace {

struct ConvOptONNXToONNXPass
    : public PassWrapper<ConvOptONNXToONNXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvOptONNXToONNXPass)

  ConvOptONNXToONNXPass() = default;
  ConvOptONNXToONNXPass(const ConvOptONNXToONNXPass &pass)
      : mlir::PassWrapper<ConvOptONNXToONNXPass,
            OperationPass<func::FuncOp>>() {}
  ConvOptONNXToONNXPass(bool enableSimdDataLayout) {
    this->enableSimdDataLayoutOpt = enableSimdDataLayout;
  };

  StringRef getArgument() const override { return "conv-opt-onnx"; }

  StringRef getDescription() const override {
    return "Perform ONNX to ONNX optimizations for optimized CPU execution of "
           "convolutions.";
  }

  // Usage: onnx-mlir-opt --conv-opt-onnx='simd-data-layout'
  Option<bool> enableSimdDataLayoutOpt{*this, "simd-data-layout",
      llvm::cl::desc("Enable SIMD data layout optimizations"),
      ::llvm::cl::init(false)};

  void runOnOperation() final;
};

void ConvOptONNXToONNXPass::runOnOperation() {
  func::FuncOp function = getOperation();
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<ONNXDialect, arith::ArithDialect, func::FuncDialect>();

  // These ops will be decomposed into other ONNX ops. Hence, they will not be
  // available after this pass.
  target.addDynamicallyLegalOp<ONNXConvOp>([&](ONNXConvOp op) {
    // Conv op can be converted to a matmul
    bool canBeAMatmul = onnx_mlir::ExpressONNXConvOpAsMatmul(op);
    bool xHasOpt = onnx_mlir::hasConvONNXTensorDataLayout(op.getX().getType());
    bool wHasOpt = onnx_mlir::hasConvONNXTensorDataLayout(op.getW().getType());
    bool hasFullOptLayout = xHasOpt && wHasOpt;
    if (DEBUG)
      fprintf(stderr,
          "ConvOps match&rewrite: went for the data simd layout opt.\n");
    bool canBeOptimized =
      canBeAMatmul || (enableSimdDataLayoutOpt && !hasFullOptLayout &&
                shouldAttemptSimdLayoutRewrite(op));
    // Conv op is legal if it cannot be further optimized.
    return !canBeOptimized;
  });

  RewritePatternSet patterns(context);
  onnx_mlir::getConvOptONNXToONNXPatterns(enableSimdDataLayoutOpt, patterns);

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    signalPassFailure();
}

} // namespace

/*!
 * Create a DecomposeONNX pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createConvOptONNXToONNXPass(
    bool enableSimdDataLayoutOpt) {
  return std::make_unique<ConvOptONNXToONNXPass>(enableSimdDataLayoutOpt);
}
