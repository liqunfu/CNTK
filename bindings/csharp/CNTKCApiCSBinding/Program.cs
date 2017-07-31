using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CNTKCApiCSBinding.CNTKObjects;
using System.Drawing;
using CNTKLibraryCSEvalExamples;

namespace CNTKCApiCSBinding
{
    class Program
    {
        static void Main(string[] args)
        {
            CNTKCApi.CNTK_SetCheckedMode(false);

            bool checked_model = CNTKCApi.CNTK_GetCheckedMode();

            string deviceKind = CNTKCApi.DeviceKindName(CNTKCApi.DeviceKind.CPU);

            // TestNDShape();
            // TestFunctionLoad();

            TestEvaluation();
        }

        static void TestEvaluation()
        {
            string modelFilePath = "ResNet20_CIFAR10_Python.model";
            CNTKFunction modelFunc = CNTKFunction.Load(modelFilePath);

            CNTKVariableVariableList inputVariables = modelFunc.Arguments();
            CNTKVariable inputVar = inputVariables.GetItem(0);

            CNTKNDShape inputShape = inputVar.Shape();
            int imageWidth = inputShape[0];
            int imageHeight = inputShape[1];
            int imageChannels = inputShape[2];
            int imageSize = inputShape.TotalSize();
            CNTKVariable outputVar = modelFunc.Output();

            CNTKUnorderedVariableValueMap inputDataMap = new CNTKUnorderedVariableValueMap();
            CNTKUnorderedVariableValueMap outputDataMap = new CNTKUnorderedVariableValueMap();

            string sampleImage = "00000.png";
            Bitmap bmp = new Bitmap(Bitmap.FromFile(sampleImage));
            var resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
            List<float> resizedCHW = resized.ParallelExtractCHW();

            CNTKListFloatWrapper cntkListFloat = new CNTKListFloatWrapper(resizedCHW);

            // Create input data map
            var inputVal = CNTKValue.CreateBatch(inputVar.Shape(), cntkListFloat);
            inputDataMap.Add(inputVar, inputVal);

            outputDataMap.Add(outputVar, null);

            modelFunc.Evaluate(inputDataMap, outputDataMap);

            CNTKValue outputVal = outputDataMap[outputVar];
            IList<IList<float>> outputData = outputVal.GetDenseDataFloat(outputVar);

            PrintOutput(outputVar.Shape().TotalSize(), outputData);
        }

        static void TestFunctionLoad()
        {
            string filePath = @"C:\LiqunWA\cntk\myfork\CNTK\Examples\Image\GettingStarted\Output\Models\01_OneHidden";
            CNTKFunction f = CNTKFunction.Load(filePath);

            IntPtr[] inputVariables = new IntPtr[100];
            f.Inputs(inputVariables);
        }

        //unsafe static void TestNDShape()
        //{
        //    int[] dim = { 2, 5 };

        //    CNTKNDShape shape = new CNTKNDShape(dim);
        //    CNTKDeviceDescriptor device = new CNTKDeviceDescriptor()
        //    {
        //        m_deviceId = 0,
        //        m_deviceType = CNTK_DeviceKind.CPU
        //    };

        //    float[] buffer = new float[2 * 5];
        //    fixed (float* p = buffer)
        //    {
        //        IntPtr ptr = (IntPtr)p;


        //        CNTKNDArrayView arrayView = new CNTKNDArrayView(
        //            CNTK_DataType.Float,
        //            shape,
        //            ptr,
        //            2 * 5 * sizeof(float),
        //            device,
        //            false);

        //        IntPtr buf = CNTKCApi.cntk_NDArrayView_WritableDataBuffer(arrayView.Handle);
        //    }
        //}

        internal static void PrintOutput<T>(int sampleSize, IList<IList<T>> outputBuffer)
        {
            Console.WriteLine("The number of sequences in the batch: " + outputBuffer.Count);
            int seqNo = 0;
            int outputSampleSize = sampleSize;
            foreach (var seq in outputBuffer)
            {
                if (seq.Count % outputSampleSize != 0)
                {
                    throw new ApplicationException("The number of elements in the sequence is not a multiple of sample size");
                }

                Console.WriteLine(String.Format("Sequence {0} contains {1} samples.", seqNo++, seq.Count / outputSampleSize));
                int i = 0;
                int sampleNo = 0;
                foreach (var element in seq)
                {
                    if (i++ % outputSampleSize == 0)
                    {
                        Console.Write(String.Format("    sample {0}: ", sampleNo));
                    }
                    Console.Write(element);
                    if (i % outputSampleSize == 0)
                    {
                        Console.WriteLine(".");
                        sampleNo++;
                    }
                    else
                    {
                        Console.Write(",");
                    }
                }
            }
        }
    }
}
