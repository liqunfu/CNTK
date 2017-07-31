using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTKLibraryCSEvalExamples
{
    public class CifarResNetTest
    {
        static Constant GetProjectionMap(uint outputDim, uint inputDim, DeviceDescriptor device)
        {
            if (inputDim > outputDim)
                throw new Exception("Can only project from lower to higher dimensionality");

            float[] projectionMapValues = new float[inputDim * outputDim];
            for (int i = 0; i < inputDim * outputDim; i++)
                projectionMapValues[i] = 0;
            for (int i = 0; i<inputDim; ++i)
                projectionMapValues[(i * (int)inputDim) + i] = 1.0f;

            var projectionMap = new NDArrayView(DataType.Float, new NDShape(new uint[]{ 1, 1, inputDim, outputDim }), device);
            projectionMap.CopyFrom(new NDArrayView(new NDShape(new uint[]{ 1, 1, inputDim, outputDim }), projectionMapValues, (uint)projectionMapValues.Count(), device));

            return new Constant(projectionMap);
        }

        static Function ResNetClassifier(Variable input, uint numOutputClasses, DeviceDescriptor device, string outputName)
        {
            double convWScale = 7.07;
            double convBValue = 0;

            double fc1WScale = 0.4;
            double fc1BValue = 0;

            double scValue = 1;
            uint bnTimeConst = 4096;

            uint kernelWidth = 3;
            uint kernelHeight = 3;

            double conv1WScale = 0.26;
            uint cMap1 = 16;
            var conv1 = CNTKLib.ConvBatchNormalizationReLULayer(input, cMap1, kernelWidth, kernelHeight, 1, 1, conv1WScale, convBValue, scValue, bnTimeConst, true /*spatial*/, device);

            var rn1_1 = CNTKLib.ResNetNode(new Variable(conv1), cMap1, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);
            var rn1_2 = CNTKLib.ResNetNode(new Variable(rn1_1), cMap1, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true /*spatial*/, device);
            var rn1_3 = CNTKLib.ResNetNode(new Variable(rn1_2), cMap1, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);

            uint cMap2 = 32;
            var rn2_1_wProj = GetProjectionMap(cMap2, cMap1, device);
            var rn2_1 = CNTKLib.ResNetNodeInc(new Variable(rn1_3), cMap2, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true /*spatial*/, rn2_1_wProj, device);
            var rn2_2 = CNTKLib.ResNetNode(new Variable(rn2_1), cMap2, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);
            var rn2_3 = CNTKLib.ResNetNode(new Variable(rn2_2), cMap2, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true /*spatial*/, device);

            uint cMap3 = 64;
            var rn3_1_wProj = GetProjectionMap(cMap3, cMap2, device);
            var rn3_1 = CNTKLib.ResNetNodeInc(new Variable(rn2_3), cMap3, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true /*spatial*/, rn3_1_wProj, device);
            var rn3_2 = CNTKLib.ResNetNode(new Variable(rn3_1), cMap3, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);
            var rn3_3 = CNTKLib.ResNetNode(new Variable(rn3_2), cMap3, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);

            // Global average pooling
            uint poolW = 8;
            uint poolH = 8;
            uint poolhStride = 1;
            uint poolvStride = 1;
            var pool = CNTKLib.Pooling(new Variable(rn3_3), PoolingType.Average, 
                new NDShape(new uint[] { poolW, poolH, 1 }), new NDShape(new uint[] { poolhStride, poolvStride, 1 }));

            // Output DNN layer
            var outTimesParams = new Parameter(new NDShape(new uint[]{ numOutputClasses, 1, 1, cMap3 }), DataType.Float, 
                CNTKLib.GlorotUniformInitializer(fc1WScale, 1, 0), device);
            var outBiasParams = new Parameter(new NDShape(new uint[] { numOutputClasses }), (float)fc1BValue, device, "");

            return CNTKLib.Plus(new Variable(CNTKLib.Times(outTimesParams, new Variable(pool))), outBiasParams, outputName);
        }

        static MinibatchSource CreateCifarMinibatchSource(uint epochSize)
        {
            int imageHeight = 32;
            int imageWidth = 32;
            int numChannels = 3;
            uint numClasses = 10;
            var mapFilePath = "train_map.txt";
            var meanFilePath = "CIFAR-10_mean.xml";

            List<Dictionary> transforms = new List<Dictionary>{
                CNTKLib.ReaderCrop("RandomSide",
                    new Tuple<int, int>(0, 0),
                    new Tuple<float, float>(0.8f, 1.0f),
                    new Tuple<float, float>(0.0f, 0.0f),
                    new Tuple<float, float>(1.0f, 1.0f),
                    "uniRatio"),
                CNTKLib.ReaderScale(imageWidth, imageHeight, numChannels),
                CNTKLib.ReaderMean(meanFilePath)
            };

            var deserializerConfiguration = CNTKLib.ImageDeserializer(mapFilePath,
                "labels", numClasses,
                "features",
                transforms);

            MinibatchSourceConfig config = new MinibatchSourceConfig(new List<Dictionary> { deserializerConfiguration });
            config.maxSamples = epochSize;

            return CNTKLib.CreateCompositeMinibatchSource(config);
        }


        public static void TrainResNetCifarClassifier(DeviceDescriptor device, bool testSaveAndReLoad)
        {
            var minibatchSource = CreateCifarMinibatchSource(MinibatchSource.InfinitelyRepeat);
            var imageStreamInfo = minibatchSource.StreamInfo("features");
            var labelStreamInfo = minibatchSource.StreamInfo("labels");

            var inputImageShape = imageStreamInfo.m_sampleLayout;
            uint numOutputClasses = (uint)(labelStreamInfo.m_sampleLayout[0]);

            var imageInputName = "Images";
            var imageInput = CNTKLib.InputVariable(inputImageShape, imageStreamInfo.m_elementType, imageInputName);
            var classifierOutput = ResNetClassifier(imageInput, numOutputClasses, device, "classifierOutput");

            var labelsInputName = "Labels";
            var labelsVar = CNTKLib.InputVariable(new NDShape(new uint[] { numOutputClasses }), labelStreamInfo.m_elementType, labelsInputName);
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(new Variable(classifierOutput), labelsVar, "lossFunction");
            var prediction = CNTKLib.ClassificationError(new Variable(classifierOutput), labelsVar, 5, "predictionError");

            if (testSaveAndReLoad)
            {
                Variable classifierOutputVar = new Variable(classifierOutput);
                Variable trainingLossVar = new Variable(trainingLoss);
                Variable predictionVar = new Variable(prediction);
                var imageClassifier = CNTKLib.Combine(new List<Variable>() { trainingLossVar, predictionVar, classifierOutputVar }, "ImageClassifier");
                TestHelper.SaveAndReloadModel(ref imageClassifier, new List<Variable> { imageInput, labelsVar, trainingLossVar, predictionVar, classifierOutputVar }, device);

                // Make sure that the names of the input variables were properly restored
                if ((imageInput.Name != imageInputName) || (labelsVar.Name != labelsInputName))
                    throw new Exception("One or more input variable names were not properly restored after save and load");

                trainingLoss = trainingLossVar;
                prediction = predictionVar.ToFunction();
                classifierOutput = classifierOutputVar.ToFunction();
            }


            TrainingParameterPerSampleScheduleDouble learningRatePerSample = new TrainingParameterPerSampleScheduleDouble(0.0078125);
            var trainer = CNTKLib.CreateTrainer(classifierOutput, trainingLoss, prediction,
                new List<Learner> { CNTKLib.SGDLearner(classifierOutput.Parameters(), learningRatePerSample) });

            const uint minibatchSize = 32;
            uint numMinibatchesToTrain = 2000;
            uint outputFrequencyInMinibatches = 20;
            for (uint i = 0; i < numMinibatchesToTrain; ++i)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(minibatchSize, device);
                trainer.TrainMinibatch(new Dictionary<Variable, MinibatchData>()
                    { { imageInput, minibatchData[imageStreamInfo] }, { labelsVar, minibatchData[labelStreamInfo] } }, device);
                TestHelper.PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
            }
        }
    }
}
