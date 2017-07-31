using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTKLibraryCSEvalExamples
{
    public class MNISTClassifierTest
    {
        public static void TrainMNISTClassifier(DeviceDescriptor device)
        {
            uint inputDim = 784;
            uint numOutputClasses = 10;
            uint hiddenLayerDim = 200;

            var input = CNTKLib.InputVariable(new NDShape(new uint[] { inputDim }), DataType.Float, "features");
            var scaledInput = CNTKLib.ElementTimes(Constant.Scalar<float>(0.00390625f, device), input);

            Function toSigmoid = TestHelper.FullyConnectedLinearLayer(new Variable(scaledInput), hiddenLayerDim, device, "");
            var classifierOutput = CNTKLib.Sigmoid(new Variable(toSigmoid), "");


            var outputTimesParam = new Parameter(NDArrayView.RandomUniform<float>(
                new NDShape(new uint[] { numOutputClasses, hiddenLayerDim }), -0.05, 0.05, 1, device));
            var outputBiasParam = new Parameter(NDArrayView.RandomUniform<float>(new NDShape(new uint[] { numOutputClasses }), -0.05, 0.05, 1, device));
            classifierOutput = CNTKLib.Plus(outputBiasParam, new Variable(CNTKLib.Times(outputTimesParam, new Variable(classifierOutput))), "classifierOutput");

            var labels = CNTKLib.InputVariable(new NDShape(new uint[] { numOutputClasses }), DataType.Float, "labels");
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(new Variable(classifierOutput), labels, "lossFunction");
            var prediction = CNTKLib.ClassificationError(new Variable(classifierOutput), labels, "classificationError");

            // Test save and reload of model
            {
                Variable classifierOutputVar = new Variable(classifierOutput);
                Variable trainingLossVar = new Variable(trainingLoss);
                Variable predictionVar = new Variable(prediction);
                var combinedNet = CNTKLib.Combine(new List<Variable>() { trainingLossVar, predictionVar, classifierOutputVar }, "MNISTClassifier");
                TestHelper.SaveAndReloadModel(ref combinedNet, new List<Variable>() { input, labels, trainingLossVar, predictionVar, classifierOutputVar }, device);

                classifierOutput = classifierOutputVar.ToFunction();
                trainingLoss = trainingLossVar.ToFunction();
                prediction = predictionVar.ToFunction();
            }

            const uint minibatchSize = 64;
            const uint numSamplesPerSweep = 60000;
            const uint numSweepsToTrainWith = 2;
            const uint numMinibatchesToTrain = (numSamplesPerSweep * numSweepsToTrainWith) / minibatchSize;

            var featureStreamName = "features";
            var labelsStreamName = "labels";
            IList<StreamConfiguration> streamConfigurations = new StreamConfiguration[]
                { new StreamConfiguration(featureStreamName, inputDim), new StreamConfiguration(labelsStreamName, numOutputClasses) };

            var minibatchSource = CNTKLib.TextFormatMinibatchSource("Train-28x28_cntk_text.txt", streamConfigurations);

            var featureStreamInfo = minibatchSource.StreamInfo(featureStreamName);
            var labelStreamInfo = minibatchSource.StreamInfo(labelsStreamName);

            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(
                0.003125, TrainingParameterScheduleDouble.UnitType.Sample);

            IList<Learner> parameterLearners = new List<Learner>() { CNTKLib.SGDLearner(classifierOutput.Parameters(), learningRatePerSample) };
            var trainer = CNTKLib.CreateTrainer(classifierOutput, trainingLoss, prediction, parameterLearners);

            uint outputFrequencyInMinibatches = 20;
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
            }
        }
    }
}
