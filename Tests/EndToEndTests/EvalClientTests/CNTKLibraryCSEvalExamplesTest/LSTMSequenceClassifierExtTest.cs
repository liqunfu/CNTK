using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTKLibraryCSEvalExamples
{
    public class LSTMSequenceClassifierExtTest
    {
        static Function LSTMSequenceClassifierNet(Variable input, uint numOutputClasses, uint embeddingDim, uint LSTMDim, uint cellDim, DeviceDescriptor device,
            string outputName)
        {
            Function embeddingFunction = CNTKLib.Embedding2(input, embeddingDim, device);
            Function LSTMFunction = CNTKLib.LSTMPComponentWithSelfStabilization2(
                embeddingFunction,
                numOutputClasses,
                LSTMDim,
                cellDim,
                device,
                "lstm");
            return LSTMFunction;
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
