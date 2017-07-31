//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// cntk_cs.i -- SWIG Interface file for C#
//

%include "CNTKManagedCommon.i"

// C# specific extenstion
%typemap(cscode) CNTK::DeviceDescriptor %{

    // Property Id.
    public int Id
    {
        get { return (int)_Id(); }
    }

    // Property Type.
    public DeviceKind Type
    {
        get { return _Type(); }
    }

    // Property CPUDevice.
    public static DeviceDescriptor CPUDevice
    {
        get { return _CPUDevice(); }
    }

    // Returns the GPUDevice with the specific deviceId.
    public static DeviceDescriptor GPUDevice(int deviceId)
    {
        if (deviceId < 0)
        {
            throw new System.ArgumentException("The paraemter deviceId should not be a negative value");
        }
        return _GPUDevice((uint)deviceId);
    }

    // Gets all devices.
    public static System.Collections.Generic.IList<DeviceDescriptor> AllDevices()
    {
        var deviceVector = _AllDevices();
        // The CopyTo is to ensure the elements in the deviceVector can live beyond deviceVector itself.
        var deviceArray = new DeviceDescriptor[deviceVector.Count];
        deviceVector.CopyTo(deviceArray);
        var deviceList = new System.Collections.Generic.List<DeviceDescriptor>(deviceArray);
        return deviceList;
    }

    // Value equality.
    public override bool Equals(System.Object obj)
    {
        // If parameter is null return false.
        if (obj == null)
        {
            return false;
        }

        // If parameter cannot be cast to Point return false.
        DeviceDescriptor p = obj as DeviceDescriptor;
        if ((System.Object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqual(this, p);
    }

    // Value equality.
    public bool Equals(DeviceDescriptor p)
    {
        // If parameter is null return false:
        if ((object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqual(this, p);
    }

    // Returns hash code value.
    public override int GetHashCode()
    {
        return this._Type().GetHashCode();
    }

    // Set devices to be excluded.
    public static void SetExcludedDevices(System.Collections.Generic.IEnumerable<DeviceDescriptor> excluded)
    {
        var excludeVector = new DeviceDescriptorVector();
        foreach (var element in excluded)
        {
            excludeVector.Add(element);
        }
        _SetExcludedDevices(excludeVector);
    }
%}


%typemap(cscode) CNTK::Axis %{

    // Property Name.
    public string Name
    {
        get { return _Name(); }
    }

    // Property IsStatic.
    public bool IsStatic
    {
        get { return _IsStaticAxis(); }
    }

    // Property IsDynamic.
    public bool IsDynamic
    {
        get { return _IsDynamicAxis(); }
    }

    // Property IsOrdered.
    public bool IsOrdered
    {
        get { return _IsOrdered(); }
    }

    // Returns index of this Axis.
    public int StaticAxisIndex(bool checkStaticAxis = true)
    {
        return _StaticAxisIndex(checkStaticAxis);
    }

    // Value equality.
    public override bool Equals(System.Object obj)
    {
        // If parameter is null return false.
        if (obj == null)
        {
            return false;
        }

        // If parameter cannot be cast to Point return false.
        Axis p = obj as Axis;
        if ((System.Object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqual(this, p);
    }

    // Value equality.
    public bool Equals(Axis p)
    {
        // If parameter is null return false:
        if ((object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqual(this, p);
    }

    // Returns hash code value.
    public override int GetHashCode()
    {
        if (this._IsDynamicAxis())
        {
            return this.Name.GetHashCode();
        }
        else
        {
            return this.StaticAxisIndex(false).GetHashCode();
        }
    }
%}

%typemap(cscode) CNTK::Function %{
%}

%typemap(cscode) CNTK::Variable %{
%}

%typemap(cscode) CNTK::NDShape %{
%}

%typemap(cscode) CNTK::NDMask %{
%}

%typemap(cscode) CNTK::Value %{
%}

%typemap(cscode) CNTK::NDArrayView %{

/* 
    public NDArrayView(DataType dataType, StorageFormat storageType, NDShape viewShape, DeviceDescriptor device)  : this()
    {
        
    }
NDArrayView(::CNTK::DataType dataType, const NDShape& viewShape, const DeviceDescriptor& device)
            : NDArrayView(dataType, StorageFormat::Dense, viewShape, device)
{}   

*/    
%}

%extend CNTK::NDArrayView {
    static NDArrayViewPtr CNTK::NDArrayView::RandomNormalFloat(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device)
    {
        return CNTK::NDArrayView::RandomNormal<float>(shape, mean, stdDev, seed, device);
    }

    static NDArrayViewPtr CNTK::NDArrayView::RandomNormalDouble(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device)
    {
        return CNTK::NDArrayView::RandomNormal<double>(shape, mean, stdDev, seed, device);
    }

    static NDArrayViewPtr CNTK::NDArrayView::RandomUniformFloat(const NDShape& shape, double rangeStart, double rangeEnd, unsigned long seed, const DeviceDescriptor& device)
    {
        return CNTK::NDArrayView::RandomNormal<float>(shape, rangeStart, rangeEnd, seed, device);
    }

    static NDArrayViewPtr CNTK::NDArrayView::RandomUniformDouble(const NDShape& shape, double rangeStart, double rangeEnd, unsigned long seed, const DeviceDescriptor& device)
    {
        return CNTK::NDArrayView::RandomNormal<double>(shape, rangeStart, rangeEnd, seed, device);
    }
}

%extend CNTK::Constant {
    static CNTK::Constant CNTK::Constant::ScalarFloat(float value, const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::CPUDevice())
    {
        return CNTK::Constant::Scalar<float>(value, device);
    }

    static CNTK::Constant CNTK::Constant::ScalarDouble(double value, const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::CPUDevice())
    {
        return CNTK::Constant::Scalar<double>(value, device);
    }
}

%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"

%template(TrainingParameterScheduleDouble) CNTK::TrainingParameterSchedule<double>;

%template(TrainingParameterPerSampleScheduleDouble) CNTK::TrainingParameterPerUnitSchedule<double, CNTK::TrainingParameterSchedule<double>::UnitType::Sample>;

%warnfilter(401, 509) CNTK::MomentumAsTimeConstantScheduleSeparatedHeader;

%include "MomentumAsTimeConstantScheduleSeparatedHeader.h"

