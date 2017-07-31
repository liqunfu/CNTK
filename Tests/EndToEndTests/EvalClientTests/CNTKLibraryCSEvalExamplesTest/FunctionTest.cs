using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTKLibraryCSEvalExamples
{
    public class FunctionTest
    {
        static IList<uint> GenerateSequenceLengths(uint numSequences, uint maxAllowedSequenceLength)
        {
            List<uint> sequenceLengths = new List<uint>((int)numSequences);
            uint maxActualSequenceLength = 0;
            uint minActualSequenceLength = 3;
            Random rnd = new Random();
            for (int i = 0; i < numSequences; ++i)
            {
                sequenceLengths.Add((uint)rnd.Next((int)minActualSequenceLength) + minActualSequenceLength);
                if (sequenceLengths[i] > maxActualSequenceLength)
                    maxActualSequenceLength = sequenceLengths[i];
            }

            return sequenceLengths;
        }

        static IList<IList<float>> GenerateSequences(IList<uint> sequenceLengths, NDShape sampleShape)
        {
            uint numSequences = (uint)sequenceLengths.Count();
            List<IList<float>> sequences = new List<IList<float>>();
            Random rnd = new Random();
            for (int i = 0; i < numSequences; ++i)
            {
                int count = sampleShape.TotalSize * (int)sequenceLengths[i];
                List<float> currentSequence = new List<float>(count);
                for (int j = 0; j < count; ++j)
                    currentSequence.Add((float)(rnd.Next(short.MaxValue)) / short.MaxValue);

                sequences.Add(currentSequence);
            }

            return sequences;
        }

        static IList<uint> GetStrides(NDShape shape)
        {
            if (shape.Rank == 0)
                return new List<uint>();

            List<uint> strides = new List<uint>(shape.Rank - 1);
            int totalSize = 1;
            for (int i = 0; i < shape.Rank - 1; ++i)
            {
                totalSize *= shape[i];
                strides.Add((uint)totalSize);
            }

            return strides;
        }

        static NDShape UnflattenedShape(uint flatennedIdx, IList<uint> strides)
        {
            NDShape unflattenedShape = new NDShape(strides.Count() + 1);
            uint remainder = flatennedIdx;
            for (int i = strides.Count() - 1; i >= 0; --i)
            {
                unflattenedShape[i + 1] = (int)(remainder / strides[i]);
                remainder = remainder % strides[i];
            }
            unflattenedShape[0] = (int)remainder;

            return unflattenedShape;
        }

        static uint FlattenedIndex(NDShape shape, IList<uint> strides)
        {
            if (shape.Rank == 0)
                return 0;

            uint flattenedIdx = (uint)shape[0];
            for (int i = 0; i < strides.Count(); ++i)
                flattenedIdx += (uint)shape[i + 1] * strides[i];

            return flattenedIdx;
        }

        const double relativeTolerance = 0.001f;
        const double absoluteTolerance = 0.000001f;

        static void FloatingPointCompare(float actual, float expected, string message)
        {
            float allowedTolerance = Math.Max((float)absoluteTolerance, Math.Abs(((float)relativeTolerance) * actual));
            if (Math.Abs(actual - expected) > allowedTolerance)
            {
                throw new Exception(message + string.Format("; Expected=%g, Actual=%g", expected, actual));
            }
        }

        static void FloatingPointCompare(double actual, double expected, string message)
        {
            double allowedTolerance = Math.Max((double)absoluteTolerance, Math.Abs(((double)relativeTolerance) * actual));
            if (Math.Abs(actual - expected) > allowedTolerance)
            {
                throw new Exception(message + string.Format("; Expected=%g, Actual=%g", expected, actual));
            }
        }

        static void FloatingPointVectorCompare(IList<float> actual, IList<float> expected, string message)
        {
            if (actual.Count() != expected.Count())
            {
                throw new Exception(message +
                    string.Format("; actual data vector size (%d) and expected data vector size (%d) are not equal", actual.Count(), expected.Count()));
            }

            for (int i = 0; i < actual.Count(); ++i)
                FloatingPointCompare(actual[i], expected[i], message);
        }

        static public void TestReduceSum(uint sampleRank, DeviceDescriptor device)
        {
            uint numSequences = 7;
            uint maxAllowedSequenceLength = 11;
            uint maxDimSize = 23;
            NDShape inputShape = new NDShape(sampleRank);
            Random rnd = new Random();
            for (uint i = 0; i < sampleRank; ++i)
            {
                inputShape.SetDimensionSize(i, (uint)rnd.Next(1, (int)maxDimSize));
            }

            var sequenceLengths = GenerateSequenceLengths(numSequences, maxAllowedSequenceLength);
            IList<IList<float>> sequences = GenerateSequences(sequenceLengths, inputShape);
            IList<bool> sequenceStartFlags = new List<bool>();
            Value sequencesValue = Value.Create(inputShape, sequences, sequenceStartFlags, device, true);

            // Test ReduceSum along a static axis
            {
                Action<int, bool> testReduceSum = (reductAxis, useNegativeAxisIndex) =>
                {
                    uint maxActualSequenceLength = (uint)sequencesValue.Shape[inputShape.Rank];
                    uint numSeq = (uint)sequencesValue.Shape[inputShape.Rank + 1];

                    var inputVar = CNTKLib.InputVariable(inputShape, DataType.Float, "input");
                    Function reduceSumFunc;

                    bool reduceAll = (reductAxis < 0);
                    if (reduceAll)
                        reduceSumFunc = CNTKLib.ReduceSum(inputVar, Axis.AllAxes());
                    else
                        reduceSumFunc = CNTKLib.ReduceSum(inputVar, new Axis(useNegativeAxisIndex ? (reductAxis - (int)sampleRank) : reductAxis));

                    NDShape outputShape = reduceSumFunc.Output.Shape;
                    NDShape outputDataShape = outputShape;
                    if (!reduceAll)
                        outputDataShape = outputDataShape.AppendShape(new NDShape(new uint[] { maxActualSequenceLength, numSeq }));

                    float[] outputData = new float[outputDataShape.TotalSize];
                    Value outputValue = new Value(new NDArrayView(outputDataShape, outputData, device, false),
                            reduceAll ? null : sequencesValue.Mask.DeepClone());

                    Dictionary<Variable, Value> outputs = new Dictionary<Variable, Value> { { reduceSumFunc.Output, outputValue } };
                    reduceSumFunc.Forward(new Dictionary<Variable, Value> { { inputVar, sequencesValue } }, outputs, device);

                    IList<uint> inputShapeStrides = GetStrides(inputShape);
                    IList<uint> outputShapeStrides = GetStrides(outputShape);

                    int totalCount = (int)(outputShape.TotalSize * maxActualSequenceLength * numSequences);
                    List<float> expectedPerFrameTotals = new List<float>();
                    for (int i = 0; i < totalCount; i++)
                    {
                        expectedPerFrameTotals.Add(0);
                    }
                    float expectedTotal = 0.0f;
                    for (int i = 0; i < numSequences; ++i)
                    {
                        int currentSequenceLength = (int)sequenceLengths[i];
                        for (int j = 0; j < currentSequenceLength; ++j)
                        {
                            for (int k = 0; k < inputShape.TotalSize; ++k)
                            {
                                var inputIdx = UnflattenedShape((uint)k, inputShapeStrides);
                                var outputIdx = inputIdx;
                                if (!reduceAll)
                                    outputIdx[reductAxis] = 0;
                                else
                                    outputIdx = new NDShape();

                                var flatOutputIdx = FlattenedIndex(outputIdx, outputShapeStrides);
                                float value = sequences[i][(j * inputShape.TotalSize) + k];
                                expectedPerFrameTotals[(((i * (int)maxActualSequenceLength) + j) * outputShape.TotalSize) + (int)flatOutputIdx] += value;
                                expectedTotal += value;
                            }
                        }
                    }

                    if (reduceAll)
                        FloatingPointVectorCompare(outputData, new List<float> { expectedTotal }, "testReduceSum: Forward prop results do not match expected results");
                    else
                        FloatingPointVectorCompare(outputData, expectedPerFrameTotals, "testReduceSum: Forward prop results do not match expected results");
                };

                // Reduce over all axes
                testReduceSum(-1, false);

                int reductionAxis = 0;
                testReduceSum(reductionAxis, true);

                if (reductionAxis < (inputShape.Rank - 1))
                    reductionAxis++;

                testReduceSum(reductionAxis, false);

                if (reductionAxis < (inputShape.Rank - 1))
                    reductionAxis++;

                testReduceSum(reductionAxis, true);
            }

            // Test ReduceSum along a dynamic axis
            {
                Action testReduceSum = () =>
                {
                    int numSeq = sequencesValue.Shape[inputShape.Rank + 1];

                    var inputVar = CNTKLib.InputVariable(inputShape, DataType.Float, "input");
                    Function reduceSumFunc = CNTKLib.SequenceReduceSum(inputVar);

                    NDShape maskShape = new NDShape(new uint[] { (uint)numSeq });
                    NDShape outputShape = reduceSumFunc.Output.Shape;
                    NDShape outputDataShape = outputShape.AppendShape(maskShape);

                    float[] outputData = new float[outputDataShape.TotalSize];
                    var maskPtr = new NDMask(maskShape, device);
                    Value outputValue = new Value(new NDArrayView(outputDataShape, outputData, device, false), maskPtr);

                    Dictionary<Variable, Value> outputs = new Dictionary<Variable, Value> { { reduceSumFunc.Output, outputValue } };
                    reduceSumFunc.Forward(new Dictionary<Variable, Value> { { inputVar, sequencesValue } }, outputs, device);

                    IList<float> expectedTotals = new List<float>(outputDataShape.TotalSize);
                    for (int i = 0; i < outputDataShape.TotalSize; i++)
                    {
                        expectedTotals.Add(0);
                    }
                    for (int i = 0; i < numSeq; ++i)
                    {
                        uint currentSequenceLength = sequenceLengths[i];
                        for (int j = 0; j < currentSequenceLength; ++j)
                        {
                            for (int k = 0; k < inputShape.TotalSize; ++k)
                            {
                                float value = sequences[i][(j * inputShape.TotalSize) + k];
                                expectedTotals[(i * inputShape.TotalSize) + k] += value;
                            }
                        }
                    }

                    FloatingPointVectorCompare(outputData, expectedTotals, "testReduceSum: Forward prop results do not match expected results");
                };

                testReduceSum();
            }
        }
    }
}
