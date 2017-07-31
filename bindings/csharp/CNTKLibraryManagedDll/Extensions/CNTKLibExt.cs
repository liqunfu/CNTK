using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK
{
    public partial class CNTKLib
    {
        static public MinibatchSource TextFormatMinibatchSource(string dataFilePath, IList<StreamConfiguration> streamConfigs,
            uint epochSize,
            bool randomize, // = true,
            uint randomizationWindow, // = MinibatchSource.DefaultRandomizationWindowInChunks,
            bool sampleBasedRandomizationWindow = false)
        {
            StreamConfigurationVector streamConfigurationVector = Helper.AsStreamConfigurationVector(streamConfigs);
            return TextFormatMinibatchSource(dataFilePath, streamConfigurationVector, epochSize, randomize,
                randomizationWindow, sampleBasedRandomizationWindow);
        }

        static public MinibatchSource TextFormatMinibatchSource(string dataFilePath, IList<StreamConfiguration> streamConfigs)
        {
            StreamConfigurationVector streamConfigurationVector = Helper.AsStreamConfigurationVector(streamConfigs);
            return TextFormatMinibatchSource(dataFilePath, streamConfigurationVector);
        }

        static public MinibatchSource TextFormatMinibatchSource(string dataFilePath, IList<StreamConfiguration> streamConfigs,
            uint epochSize)
        {
            StreamConfigurationVector streamConfigurationVector = Helper.AsStreamConfigurationVector(streamConfigs);
            return TextFormatMinibatchSource(dataFilePath, streamConfigurationVector, epochSize);
        }

        static public MinibatchSource TextFormatMinibatchSource(string dataFilePath, IList<StreamConfiguration> streamConfigs,
            uint epochSize, bool randomize)
        {
            StreamConfigurationVector streamConfigurationVector = Helper.AsStreamConfigurationVector(streamConfigs);
            return TextFormatMinibatchSource(dataFilePath, streamConfigurationVector, epochSize, randomize);
        }

        static public MinibatchSource TextFormatMinibatchSourceExperiment(string dataFilePath, IList<StreamConfiguration> streamConfigs)
        {
            StreamConfigurationVector streamConfigurationVector = Helper.AsStreamConfigurationVector(streamConfigs);
            return TextFormatMinibatchSourceExperiment(dataFilePath, streamConfigurationVector);
        }

        static public MinibatchSource TextFormatMinibatchSourceExperimentRandom(string dataFilePath, IList<StreamConfiguration> streamConfigs)
        {
            StreamConfigurationVector streamConfigurationVector = Helper.AsStreamConfigurationVector(streamConfigs);
            return TextFormatMinibatchSourceExperimentRandom(dataFilePath, streamConfigurationVector);
        }

        public static void ComputeInputPerDimMeansAndInvStdDevs(MinibatchSource minibatchSource,
            IDictionary<StreamInformation, Tuple<NDArrayView, NDArrayView>> computedMeanAndVariances,
            DeviceDescriptor device)
        {
            UnorderedMapStreamInformationPairNDArrayViewPtrNDArrayViewPtr mapStreamInfoToNDArrayPair =
                Helper.AsUnorderedMapStreamInformationPairNDArrayViewPtrNDArrayViewPtr(computedMeanAndVariances);
            ComputeInputPerDimMeansAndInvStdDevs(minibatchSource, mapStreamInfoToNDArrayPair, device);

            foreach (StreamInformation s in computedMeanAndVariances.Keys.ToList())
            {
                computedMeanAndVariances[s] = new Tuple<NDArrayView, NDArrayView>(mapStreamInfoToNDArrayPair[s].first, mapStreamInfoToNDArrayPair[s].second);
            }
        }

        public static Learner SGDLearner(IList<Parameter> parameters, TrainingParameterScheduleDouble learningRateSchedule, AdditionalLearningOptions additionalOptions)
        {
            ParameterVector parameterVector = Helper.AsParameterVector(parameters);
            return SGDLearner(parameterVector, learningRateSchedule, additionalOptions);
        }

        public static Trainer CreateTrainer(Function model, Function lossFunction, Function evaluationFunction, IList<Learner> parameterLearners,
            ProgressWriterVector progressWriters = null)
        {
            LearnerVector learnerVector = Helper.AsLearnerVector(parameterLearners);
            if (progressWriters != null)
                return CreateTrainer(model, lossFunction, evaluationFunction, learnerVector, progressWriters);
            else
                return CreateTrainer(model, lossFunction, evaluationFunction, learnerVector);
        }

        public static Function Combine(IList<Variable> operands, string name)
        {
            VariableVector operandVector = Helper.AsVariableVector(operands);
            return Combine(operandVector, name);
        }

        public static Variable InputVariable(NDShape shape, bool isSparse, DataType dataType, bool needsGradient, string name, IList<Axis> dynamicAxes)
        {
            AxisVector dynamicAxesVector = Helper.AsAxisVector(dynamicAxes);
            return InputVariable(shape, isSparse, dataType, needsGradient, name, dynamicAxesVector);
        }

        public static Variable InputVariable(NDShape shape, bool isSparse, DataType dataType, string name, IList<Axis> dynamicAxes)
        {
            AxisVector dynamicAxesVector = Helper.AsAxisVector(dynamicAxes);
            return InputVariable(shape, isSparse, dataType, name, dynamicAxesVector);
        }

        public static Variable PlaceholderVariable(NDShape shape, IList<Axis> dynamicAxes)
        {
            AxisVector dynamicAxesVector = Helper.AsAxisVector(dynamicAxes);
            return PlaceholderVariable(shape, dynamicAxesVector);
        }
        public static Dictionary ReaderCrop(string cropType, Tuple<int, int> cropSize, Tuple<float, float> sideRatio,
            Tuple<float, float> areaRatio, Tuple<float, float> aspectRatio, string jitterType)
        {
            PairIntInt cropSizeSwig = new PairIntInt(cropSize.Item1, cropSize.Item2);
            PairFloatFloat sideRatioSwig = new PairFloatFloat(sideRatio.Item1, sideRatio.Item2);
            PairFloatFloat areaRatioSwig = new PairFloatFloat(areaRatio.Item1, areaRatio.Item2);
            PairFloatFloat aspectRatioSwig = new PairFloatFloat(aspectRatio.Item1, aspectRatio.Item2);
            return ReaderCrop(cropType, cropSizeSwig, sideRatioSwig, areaRatioSwig, aspectRatioSwig, jitterType);
        }

        public static Dictionary ImageDeserializer(string fileName, string labelStreamName, uint numLabels, string imageStreamName, IList<Dictionary> deserializers)
        {
            DictionaryVector deserializersSwig = Helper.AsDictionaryVector(deserializers);
            return ImageDeserializer(fileName, labelStreamName, numLabels, imageStreamName, deserializersSwig);
        }
    }
}
