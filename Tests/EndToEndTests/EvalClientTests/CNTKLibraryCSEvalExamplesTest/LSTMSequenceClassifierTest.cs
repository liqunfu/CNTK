using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTKLibraryCSEvalExamples
{
    public class LSTMSequenceClassifierTest
    {
        static Function Stabilize<ElementType>(Variable x, DeviceDescriptor device)
        {
            bool isFloatType = typeof(ElementType).Equals(typeof(float));
            Constant f, fInv;
            if (isFloatType)
            {
                f = Constant.Scalar(4.0f, device);
                fInv = Constant.Scalar(f.DataType, 1.0 / 4.0f);
            }
            else
            {
                f = Constant.Scalar(4.0, device);
                fInv = Constant.Scalar(f.DataType, 1.0 / 4.0f);
            }

            var beta = CNTKLib.ElementTimes(
                fInv,
                CNTKLib.Log( 
                    Constant.Scalar(f.DataType, 1.0) +  
                    CNTKLib.Exp(CNTKLib.ElementTimes(f, new Parameter(new NDShape(), f.DataType, 0.99537863 /* 1/f*ln (e^f-1) */, device)))));
            return CNTKLib.ElementTimes(beta, x);
        }

        static Tuple<Function, Function> LSTMPCellWithSelfStabilization<ElementType>( 
            Variable input, Variable prevOutput, Variable prevCellState, DeviceDescriptor device)
        {
            uint outputDim = (uint)prevOutput.Shape[0];
            uint cellDim = (uint)prevCellState.Shape[0];

            bool isFloatType = typeof(ElementType).Equals(typeof(float));
            DataType dataType = isFloatType ? DataType.Float : DataType.Double;

            // new Parameter(new NDShape(new uint[] { 1 }), (ElementType)(object)0.0, device, "");
            // TODO, how to use ElementType?
            Func<uint, Parameter> createBiasParam;
            if (isFloatType)
                createBiasParam = (dim) => new Parameter(new uint[] { dim }, 0.01f, device, "");
            else
                createBiasParam = (dim) => new Parameter(new uint[] { dim }, 0.01, device, "");

            uint seed2 = 1;
            Func<uint, Parameter> createProjectionParam = (oDim) => new Parameter(new uint[] { oDim, (uint)NDShape.InferredDimension },
                    dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed2++), device);

            Func<uint, Parameter> createDiagWeightParam = (dim) =>
                new Parameter(new uint[] { dim }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed2++), device);

            Function stabilizedPrevOutput = Stabilize<ElementType>(prevOutput, device);
            Function stabilizedPrevCellState = Stabilize<ElementType>(prevCellState, device);

            Func<Variable> projectInput = () =>
                createBiasParam(cellDim) + (createProjectionParam(cellDim) * input);

            // Input gate
            Function it =
                CNTKLib.Sigmoid(
                    (Variable)(projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)) +  
                    CNTKLib.ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
            Function bit = CNTKLib.ElementTimes(
                it,
                CNTKLib.Tanh(projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)));

            // Forget-me-not gate
            Function ft = CNTKLib.Sigmoid(
                (Variable)(
                        projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)) +
                        CNTKLib.ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
            Function bft = CNTKLib.ElementTimes(ft, prevCellState);

            Function ct = (Variable)bft + bit;

            // Output gate
            Function ot = CNTKLib.Sigmoid( 
                (Variable)(projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)) + 
                CNTKLib.ElementTimes(createDiagWeightParam(cellDim), Stabilize<ElementType>(ct, device)));
            Function ht = CNTKLib.ElementTimes(ot, CNTKLib.Tanh(ct));

            Function c = ct;
            Function h = (outputDim != cellDim) ? (createProjectionParam(outputDim) * Stabilize<ElementType>(ht, device)) : ht;

            return new Tuple<Function, Function>(h, c);
        }


        static Tuple<Function, Function> LSTMPComponentWithSelfStabilization<ElementType>(Variable input,
            NDShape outputShape, NDShape cellShape,
            Func<Variable, Function> recurrenceHookH,
            Func<Variable, Function> recurrenceHookC,
            DeviceDescriptor device)
        {
            var dh = CNTKLib.PlaceholderVariable(outputShape, input.DynamicAxes);
            var dc = CNTKLib.PlaceholderVariable(cellShape, input.DynamicAxes);

            var LSTMCell = LSTMPCellWithSelfStabilization<ElementType>(input, dh, dc, device);
            var actualDh = recurrenceHookH(LSTMCell.Item1);
            var actualDc = recurrenceHookC(LSTMCell.Item2);

            // Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
            LSTMCell.Item1.ReplacePlaceholders(new Dictionary<Variable, Variable> { { dh, actualDh }, { dc, actualDc } });

            return new Tuple<Function, Function>(LSTMCell.Item1, LSTMCell.Item2);
        }



        private static Function Embedding(Variable input, uint embeddingDim, DeviceDescriptor device)
        {
            System.Diagnostics.Debug.Assert(input.Shape.Rank == 1);
            uint inputDim = (uint)input.Shape[0];
            var embeddingParameters = new Parameter(new uint[] { embeddingDim, inputDim }, DataType.Float, CNTKLib.GlorotUniformInitializer(), device);
            return CNTKLib.Times(embeddingParameters, input);
        }

        static Function LSTMSequenceClassifierNet(Variable input, uint numOutputClasses, uint embeddingDim, uint LSTMDim, uint cellDim, DeviceDescriptor device, 
            string outputName)
        {
            Function embeddingFunction = Embedding(input, embeddingDim, device);
            Func<Variable, Function> pastValueRecurrenceHook = (x) => CNTKLib.PastValue(x);
            Function LSTMFunction = LSTMPComponentWithSelfStabilization<float>(
                embeddingFunction,
                new uint[] { LSTMDim },
                new uint[] { cellDim },
                pastValueRecurrenceHook,
                pastValueRecurrenceHook,
                device).Item1;
            Function thoughtVectorFunction = CNTKLib.Last(LSTMFunction);

            return TestHelper.FullyConnectedLinearLayer(thoughtVectorFunction, numOutputClasses, device, outputName);
        }

        public static void TrainLSTMSequenceClassifier(DeviceDescriptor device, bool useSparseLabels, bool testSaveAndReLoad)
        {
            const uint inputDim = 2000;
            const uint cellDim = 25;
            const uint hiddenDim = 25;
            const uint embeddingDim = 50;
            const uint numOutputClasses = 5;

            var featuresName = "features";
            var features = CNTKLib.InputVariable(new uint[] { inputDim }, true /*isSparse*/, DataType.Float, featuresName);
            var classifierOutput = LSTMSequenceClassifierNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, "classifierOutput");

            var labelsName = "labels";
            var labels = CNTKLib.InputVariable(new uint[] { numOutputClasses }, useSparseLabels, DataType.Float, labelsName,
                new List<Axis>() { Axis.DefaultBatchAxis() });
            Function trainingLoss = CNTKLib.CrossEntropyWithSoftmax(classifierOutput, labels, "lossFunction");
            Function prediction = CNTKLib.ClassificationError(classifierOutput, labels, "classificationError");

            if (testSaveAndReLoad)
            {
                Variable classifierOutputVar = classifierOutput;
                Variable trainingLossVar = trainingLoss;
                Variable predictionVar = prediction;
                var oneHiddenLayerClassifier = CNTKLib.Combine(new List<Variable>() { trainingLossVar, predictionVar, classifierOutputVar }, "classifierModel");
                TestHelper.SaveAndReloadModel(ref oneHiddenLayerClassifier, new List<Variable>() { features, labels, trainingLossVar, predictionVar, classifierOutputVar }, device);

                // NOT NEEDED
                classifierOutput = classifierOutputVar;
                trainingLoss = trainingLossVar;
                prediction = predictionVar;
            }

            IList<StreamConfiguration> streamConfigurations = new StreamConfiguration[]
                { new StreamConfiguration(featuresName, inputDim, true, "x"), new StreamConfiguration(labelsName, numOutputClasses, false, "y") };

            // TODO:
            var minibatchSource = CNTKLib.TextFormatMinibatchSourceExperimentRandom("Train.ctf", streamConfigurations);

            const uint minibatchSize = 200;

            var featureStreamInfo = minibatchSource.StreamInfo(featuresName);
            var labelStreamInfo = minibatchSource.StreamInfo(labelsName);

            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(
                0.0005, TrainingParameterScheduleDouble.UnitType.Sample);

            CNTK.MomentumAsTimeConstantScheduleSeparatedHeader momentumTimeConstant = new MomentumAsTimeConstantScheduleSeparatedHeader(256);

            IList<Learner> parameterLearners = new List<Learner>() {
                CNTKLib.MomentumSGDLearner(classifierOutput.Parameters(), learningRatePerSample, momentumTimeConstant, /*unitGainMomentum = */true)  };
            var trainer = CNTKLib.CreateTrainer(classifierOutput, trainingLoss, prediction, parameterLearners);

            uint outputFrequencyInMinibatches = 1;
            for (uint i = 0; true; i++)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(minibatchSize, device);
                if (minibatchData.empty())
                    break;

                var arguments = new Dictionary<Variable, MinibatchData>
                {
                    { features, minibatchData[featureStreamInfo] },
                    { labels, minibatchData[labelStreamInfo] }
                };

                trainer.TrainMinibatch(arguments, device);
                TestHelper.PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
            }
        }
    }
}
