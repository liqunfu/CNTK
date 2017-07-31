#include "CNTKLibrary.h"

namespace CNTK
{
    namespace Internal
    {
        template <typename ElementType>
        inline FunctionPtr Stabilize(const Variable& x, const DeviceDescriptor& device)
        {
            ElementType scalarConstant = 4.0f;
            auto f = Constant::Scalar(scalarConstant);
            auto fInv = Constant::Scalar(f.GetDataType(), 1.0 / scalarConstant);

            auto beta = ElementTimes(fInv, Log(Constant::Scalar(f.GetDataType(), 1.0) + Exp(ElementTimes(f, Parameter({}, f.GetDataType(), 0.99537863 /* 1/f*ln (e^f-1) */, device)))));
            return ElementTimes(beta, x);
        }

        template <typename ElementType>
        std::pair<FunctionPtr, FunctionPtr> LSTMPCellWithSelfStabilization(Variable input, Variable prevOutput, Variable prevCellState, const DeviceDescriptor& device)
        {
            size_t outputDim = prevOutput.Shape()[0];
            size_t cellDim = prevCellState.Shape()[0];

            auto createBiasParam = [device](size_t dim) {
                return Parameter({ dim }, (ElementType)0.0, device);
            };

            unsigned long seed2 = 1;
            auto createProjectionParam = [device, &seed2](size_t outputDim) {
                return Parameter({ outputDim, NDShape::InferredDimension }, AsDataType<ElementType>(), GlorotUniformInitializer(1.0, 1, 0, seed2++), device);
            };

            auto createDiagWeightParam = [device, &seed2](size_t dim) {
                return Parameter({ dim }, AsDataType<ElementType>(), GlorotUniformInitializer(1.0, 1, 0, seed2++), device);
            };

            auto stabilizedPrevOutput = Stabilize<ElementType>(prevOutput, device);
            auto stabilizedPrevCellState = Stabilize<ElementType>(prevCellState, device);

            auto projectInput = [input, cellDim, createBiasParam, createProjectionParam]() {
                return createBiasParam(cellDim) + Times(createProjectionParam(cellDim), input);
            };

            // Input gate
            auto it = Sigmoid(projectInput() + Times(createProjectionParam(cellDim), stabilizedPrevOutput) + ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
            auto bit = ElementTimes(it, Tanh(projectInput() + Times(createProjectionParam(cellDim), stabilizedPrevOutput)));

            // Forget-me-not gate
            auto ft = Sigmoid(projectInput() + Times(createProjectionParam(cellDim), stabilizedPrevOutput) + ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
            auto bft = ElementTimes(ft, prevCellState);

            auto ct = bft + bit;

            // Output gate
            auto ot = Sigmoid(projectInput() + Times(createProjectionParam(cellDim), stabilizedPrevOutput) + ElementTimes(createDiagWeightParam(cellDim), Stabilize<ElementType>(ct, device)));
            auto ht = ElementTimes(ot, Tanh(ct));

            auto c = ct;
            auto h = (outputDim != cellDim) ? Times(createProjectionParam(outputDim), Stabilize<ElementType>(ht, device)) : ht;

            return{ h, c };
        }

        template <typename ElementType>
        std::pair<FunctionPtr, FunctionPtr> LSTMPComponentWithSelfStabilization(Variable input,
            const NDShape& outputShape,
            const NDShape& cellShape,
            const std::function<FunctionPtr(const Variable&)>& recurrenceHookH,
            const std::function<FunctionPtr(const Variable&)>& recurrenceHookC,
            const DeviceDescriptor& device)
        {
            auto dh = PlaceholderVariable(outputShape, input.DynamicAxes());
            auto dc = PlaceholderVariable(cellShape, input.DynamicAxes());

            auto LSTMCell = LSTMPCellWithSelfStabilization<ElementType>(input, dh, dc, device);

            auto actualDh = recurrenceHookH(LSTMCell.first);
            auto actualDc = recurrenceHookC(LSTMCell.second);

            // Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
            LSTMCell.first->ReplacePlaceholders({ { dh, actualDh },{ dc, actualDc } });

            return{ LSTMCell.first, LSTMCell.second };
        }

        inline FunctionPtr FullyConnectedLinearLayer(Variable input, size_t outputDim, const DeviceDescriptor& device, const std::wstring& outputName = L"")
        {
            assert(input.Shape().Rank() == 1);
            size_t inputDim = input.Shape()[0];

            auto timesParam = Parameter({ outputDim, inputDim }, DataType::Float, GlorotUniformInitializer(DefaultParamInitScale, SentinelValueForInferParamInitRank, SentinelValueForInferParamInitRank, 1), device, L"timesParam");
            auto timesFunction = Times(timesParam, input, L"times");

            auto plusParam = Parameter({ outputDim }, 0.0f, device, L"plusParam");
            return Plus(plusParam, timesFunction, outputName);
        }
    }

    using namespace Internal;

    FunctionPtr Embedding2(const Variable& input, size_t embeddingDim, const DeviceDescriptor& device)
    {
        assert(input.Shape().Rank() == 1);
        size_t inputDim = input.Shape()[0];
        auto embeddingParameters = Parameter({ embeddingDim, inputDim }, DataType::Float, GlorotUniformInitializer(), device);
        return Times(embeddingParameters, input);
    }

    FunctionPtr LSTMPComponentWithSelfStabilization2(const Variable& embeddingFunction, size_t numOutputClasses,
        size_t LSTMDim, size_t cellDim, const DeviceDescriptor& device, const std::wstring& outputName)
    {
        auto pastValueRecurrenceHook = [](const Variable& x) { return PastValue(x); };
        auto LSTMFunction = LSTMPComponentWithSelfStabilization<float>(
            embeddingFunction, { LSTMDim }, { cellDim }, pastValueRecurrenceHook, pastValueRecurrenceHook, device).first;
        auto thoughtVectorFunction = Sequence::Last(LSTMFunction);

        return FullyConnectedLinearLayer(thoughtVectorFunction, numOutputClasses, device, outputName);
    }

    inline FunctionPtr ConvBatchNormalizationLayer(Variable input, size_t outFeatureMapCount, size_t kernelWidth, size_t kernelHeight, size_t hStride, size_t vStride, double wScale, double bValue, double scValue, size_t bnTimeConst, bool spatial, const DeviceDescriptor& device)
    {
        size_t numInputChannels = input.Shape()[input.Shape().Rank() - 1];

        auto convParams = Parameter({ kernelWidth, kernelHeight, numInputChannels, outFeatureMapCount }, DataType::Float, GlorotUniformInitializer(wScale, -1, 2), device);
        auto convFunction = Convolution(convParams, input, { hStride, vStride, numInputChannels });

        auto biasParams = Parameter({ NDShape::InferredDimension }, (float)bValue, device);
        auto scaleParams = Parameter({ NDShape::InferredDimension }, (float)scValue, device);
        auto runningMean = Constant({ NDShape::InferredDimension }, 0.0f, device);
        auto runningInvStd = Constant({ NDShape::InferredDimension }, 0.0f, device);
        auto runningCount = Constant::Scalar(0.0f, device);
        return BatchNormalization(convFunction, scaleParams, biasParams, runningMean, runningInvStd, runningCount, spatial, (double)bnTimeConst, 0.0, 1e-5 /* epsilon */);
    }

    inline FunctionPtr ConvBatchNormalizationReLULayer(Variable input, size_t outFeatureMapCount, size_t kernelWidth, size_t kernelHeight, size_t hStride, size_t vStride, double wScale, double bValue, double scValue, size_t bnTimeConst, bool spatial, const DeviceDescriptor& device)
    {
        auto convBNFunction = ConvBatchNormalizationLayer(input, outFeatureMapCount, kernelWidth, kernelHeight, hStride, vStride, wScale, bValue, scValue, bnTimeConst, spatial, device);
        return ReLU(convBNFunction);
    }

    inline FunctionPtr ProjectLayer(Variable wProj, Variable input, size_t hStride, size_t vStride, double bValue, double scValue, size_t bnTimeConst, const DeviceDescriptor& device)
    {
        size_t outFeatureMapCount = wProj.Shape()[0];
        auto b = Parameter({ outFeatureMapCount }, (float)bValue, device);
        auto sc = Parameter({ outFeatureMapCount }, (float)scValue, device);
        auto m = Constant({ outFeatureMapCount }, 0.0f, device);
        auto v = Constant({ outFeatureMapCount }, 0.0f, device);

        auto n = Constant::Scalar(0.0f, device);

        size_t numInputChannels = input.Shape()[input.Shape().Rank() - 1];

        auto c = Convolution(wProj, input, { hStride, vStride, numInputChannels }, { true }, { false });
        return BatchNormalization(c, sc, b, m, v, n, true /*spatial*/, (double)bnTimeConst, 0, 1e-5, false); // TODO: cudnn engine does not work in Linux debug build here
    }

    inline FunctionPtr ResNetNode(Variable input, size_t outFeatureMapCount, size_t kernelWidth, size_t kernelHeight, double wScale, double bValue, double scValue, size_t bnTimeConst, bool spatial, const DeviceDescriptor& device)
    {
        auto c1 = ConvBatchNormalizationReLULayer(input, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device);
        auto c2 = ConvBatchNormalizationLayer(c1, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device);
        auto p = Plus(c2, input);
        return ReLU(p);
    }

    inline FunctionPtr ResNetNodeInc(Variable input, size_t outFeatureMapCount, size_t kernelWidth, size_t kernelHeight, double wScale, double bValue, double scValue, size_t bnTimeConst, bool spatial, Variable wProj, const DeviceDescriptor& device)
    {
        auto c1 = ConvBatchNormalizationReLULayer(input, outFeatureMapCount, kernelWidth, kernelHeight, 2, 2, wScale, bValue, scValue, bnTimeConst, spatial, device);
        auto c2 = ConvBatchNormalizationLayer(c1, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device);

        auto cProj = ProjectLayer(wProj, input, 2, 2, bValue, scValue, bnTimeConst, device);

        auto p = Plus(c2, cProj);
        return ReLU(p);
    }

    // Standard building block for ResNet with identity shortcut(option A).
    inline FunctionPtr ResNetNodeA(Variable input, size_t outFeatureMapCount, size_t kernelWidth, size_t kernelHeight, double wScale, double bValue, double scValue, size_t bnTimeConst, bool spatial, const DeviceDescriptor& device)
    {
        auto conv1 = ConvBatchNormalizationReLULayer(input, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device);
        auto conv2 = ConvBatchNormalizationLayer(conv1, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device);

        // Identity shortcut followed by ReLU.
        return ReLU(Plus(conv2, input));
    }

    // Standard building block for ResNet with padding(option B).
    inline FunctionPtr ResNetNodeBInc(Variable input, size_t outFeatureMapCount, size_t kernelWidth, size_t kernelHeight, double wScale, double bValue, double scValue, size_t bnTimeConst, bool spatial, const DeviceDescriptor& device)
    {
        auto conv1 = ConvBatchNormalizationReLULayer(input, outFeatureMapCount, kernelWidth, kernelHeight, 2, 2, wScale, bValue, scValue, bnTimeConst, spatial, device);
        auto conv2 = ConvBatchNormalizationLayer(conv1, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device);

        // Projection convolution layer.
        auto cProj = ConvBatchNormalizationLayer(input, outFeatureMapCount, 1, 1, 2, 2, wScale, bValue, scValue, bnTimeConst, spatial, device);
        return ReLU(Plus(conv2, cProj));
    }
}