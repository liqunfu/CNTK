using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTKLibraryCSEvalExamples
{
    public class SimpleFeedForwardClassifierTest
    {
        public static void TrainSimpleFeedForwardClassifier(DeviceDescriptor device)
        {
            uint inputDim = 2;
            uint numOutputClasses = 2;
            uint hiddenLayerDim = 50;
            uint numHiddenLayers = 2;

            uint minibatchSize = 50;
            uint numSamplesPerSweep = 10000;
            uint numSweepsToTrainWith = 2;
            uint numMinibatchesToTrain = (numSamplesPerSweep * numSweepsToTrainWith) / minibatchSize;

            var featureStreamName = "features";
            var labelsStreamName = "labels";
            IList<StreamConfiguration> streamConfigurations = new StreamConfiguration[]
                { new StreamConfiguration(featureStreamName, inputDim), new StreamConfiguration(labelsStreamName, numOutputClasses) };

            // TODO:
            var minibatchSource = CNTKLib.TextFormatMinibatchSourceExperiment("SimpleDataTrain_cntk_text.txt", streamConfigurations);
            var featureStreamInfo = minibatchSource.StreamInfo(featureStreamName);
            var labelStreamInfo = minibatchSource.StreamInfo(labelsStreamName);

            IDictionary<StreamInformation, Tuple<NDArrayView, NDArrayView>> inputMeansAndInvStdDevs =
                new Dictionary<StreamInformation, Tuple<NDArrayView, NDArrayView>> { { featureStreamInfo, new Tuple<NDArrayView, NDArrayView>(null, null) } };
            CNTKLib.ComputeInputPerDimMeansAndInvStdDevs(minibatchSource, inputMeansAndInvStdDevs, device);

            var input = CNTKLib.InputVariable(new NDShape(new uint[] { inputDim }), DataType.Float, "features");
            var normalizedinput = CNTKLib.PerDimMeanVarianceNormalize(input, inputMeansAndInvStdDevs[featureStreamInfo].Item1, inputMeansAndInvStdDevs[featureStreamInfo].Item2);
            Function toSigmoid = TestHelper.FullyConnectedLinearLayer(normalizedinput, hiddenLayerDim, device, "");
            var classifierOutput = CNTKLib.Sigmoid(toSigmoid, "");

            for (uint i = 1; i < numHiddenLayers; ++i)
            {
                Function toSigmoid2 = TestHelper.FullyConnectedLinearLayer(classifierOutput, hiddenLayerDim, device, "");
                classifierOutput = CNTKLib.Sigmoid(toSigmoid2, "");
            }

            var outputTimesParam = new Parameter(NDArrayView.RandomUniform<float>(new NDShape(new uint[] { numOutputClasses, hiddenLayerDim }), -0.05, 0.05, 1, device));
            var outputBiasParam = new Parameter(NDArrayView.RandomUniform<float>(new NDShape(new uint[] { numOutputClasses }), -0.05, 0.05, 1, device));
            classifierOutput = CNTKLib.Plus(outputBiasParam, outputTimesParam * classifierOutput, "classifierOutput");

            var labels = CNTKLib.InputVariable(new NDShape(new uint[] { numOutputClasses }), DataType.Float, "labels");
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(classifierOutput, labels, "lossFunction"); ;
            var prediction = CNTKLib.ClassificationError(classifierOutput, labels, "classificationError");

            // Test save and reload of model
            {
                Variable classifierOutputVar = classifierOutput;
                Variable trainingLossVar = trainingLoss;
                Variable predictionVar = prediction;
                var combinedNet = CNTKLib.Combine(new List<Variable>() { trainingLoss, prediction, classifierOutput },
                    "feedForwardClassifier");
                TestHelper.SaveAndReloadModel(ref combinedNet, new List<Variable>() { input, labels, trainingLossVar, predictionVar, classifierOutputVar }, device);

                classifierOutput = classifierOutputVar;
                trainingLoss = trainingLossVar;
                prediction = predictionVar;
            }

            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(
                0.02, TrainingParameterScheduleDouble.UnitType.Sample);

            IList<StreamConfiguration> streamConfigurations2 = new StreamConfiguration[]
                { new StreamConfiguration("features", inputDim), new StreamConfiguration("labels", numOutputClasses) };
            minibatchSource = CNTKLib.TextFormatMinibatchSource("SimpleDataTrain_cntk_text.txt", streamConfigurations2);
            IList<Learner> parameterLearners = new List<Learner>() { CNTKLib.SGDLearner(classifierOutput.Parameters(), learningRatePerSample) };
            var trainer = CNTKLib.CreateTrainer(classifierOutput, trainingLoss, prediction, parameterLearners);
            uint outputFrequencyInMinibatches = 20;
            uint trainingCheckpointFrequency = 100;
            for (uint i = 0; i < numMinibatchesToTrain; ++i)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(minibatchSize, device);
                var arguments = new Dictionary<Variable, MinibatchData>
                {
                    { input, minibatchData[featureStreamInfo] },
                    { labels, minibatchData[labelStreamInfo] }
                };
                trainer.TrainMinibatch(arguments, device);
                TestHelper.PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);

                if ((i % trainingCheckpointFrequency) == (trainingCheckpointFrequency - 1))
                {
                    string ckpName = "feedForward.net";
                    trainer.SaveCheckpoint(ckpName);
                    trainer.RestoreFromCheckpoint(ckpName);
                }
            }
        }

    }
}
