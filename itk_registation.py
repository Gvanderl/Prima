import argparse
import itk
import cv2
from baseline import grayscale, whitewashing
import numpy as np

# if VS(itk.Version.GetITKVersion()) < VS("4.9.0"):
#     print("ITK 4.9.0 is required.")
#     sys.exit(1)

parser = argparse.ArgumentParser(
    description="Perform 2D Translation Registration With Mean Squares."
)
parser.add_argument("fixed_input_image")
parser.add_argument("moving_input_image")
parser.add_argument("output_image")
parser.add_argument("difference_image_after")
parser.add_argument("difference_image_before")
args = parser.parse_args()

PixelType = itk.ctype("float")

fixedImage = itk.imread(args.fixed_input_image, PixelType)
movingImage = itk.imread(args.moving_input_image, PixelType)

# fixedImage, movingImage = whitewashing(fixedImage), whitewashing(movingImage)
# fixedImage, movingImage = itk.image_from_array(fixedImage), itk.image_from_array(movingImage)

print(1)
Dimension = fixedImage.GetImageDimension()
FixedImageType = itk.Image[PixelType, Dimension]
MovingImageType = itk.Image[PixelType, Dimension]
print(2)
TransformType = itk.TranslationTransform[itk.D, Dimension]
initialTransform = TransformType.New()
print(3)
optimizer = itk.RegularStepGradientDescentOptimizerv4.New(
    LearningRate=4,
    MinimumStepLength=0.001,
    RelaxationFactor=0.5,
    NumberOfIterations=200,
)

metric = itk.MeanSquaresImageToImageMetricv4[FixedImageType, MovingImageType].New()
print(4)
registration = itk.ImageRegistrationMethodv4.New(
    FixedImage=fixedImage,
    MovingImage=movingImage,
    Metric=metric,
    Optimizer=optimizer,
    InitialTransform=initialTransform,
)
print(5)
movingInitialTransform = TransformType.New()
initialParameters = movingInitialTransform.GetParameters()
initialParameters[0] = 0
initialParameters[1] = 0
movingInitialTransform.SetParameters(initialParameters)
registration.SetMovingInitialTransform(movingInitialTransform)
print(6)
identityTransform = TransformType.New()
identityTransform.SetIdentity()
registration.SetFixedInitialTransform(identityTransform)

registration.SetNumberOfLevels(1)
registration.SetSmoothingSigmasPerLevel([0])
registration.SetShrinkFactorsPerLevel([1])
print(7)
registration.Update()

transform = registration.GetTransform()
finalParameters = transform.GetParameters()
translationAlongX = finalParameters.GetElement(0)
translationAlongY = finalParameters.GetElement(1)
print(8)
numberOfIterations = optimizer.GetCurrentIteration()

bestValue = optimizer.GetValue()

print("Result = ")
print(" Translation X = " + str(translationAlongX))
print(" Translation Y = " + str(translationAlongY))
print(" Iterations    = " + str(numberOfIterations))
print(" Metric value  = " + str(bestValue))

CompositeTransformType = itk.CompositeTransform[itk.D, Dimension]
outputCompositeTransform = CompositeTransformType.New()
outputCompositeTransform.AddTransform(movingInitialTransform)
outputCompositeTransform.AddTransform(registration.GetModifiableTransform())

resampler = itk.ResampleImageFilter.New(
    Input=movingImage,
    Transform=outputCompositeTransform,
    UseReferenceImage=True,
    ReferenceImage=fixedImage,
)
resampler.SetDefaultPixelValue(100)

OutputPixelType = itk.ctype("unsigned char")
OutputImageType = itk.Image[OutputPixelType, Dimension]

caster = itk.CastImageFilter[FixedImageType, OutputImageType].New(Input=resampler)

writer = itk.ImageFileWriter.New(Input=caster, FileName=args.output_image)
writer.Update()

difference = itk.SubtractImageFilter.New(Input1=fixedImage, Input2=resampler)

intensityRescaler = itk.RescaleIntensityImageFilter[
    FixedImageType, OutputImageType
].New(
    Input=difference,
    OutputMinimum=itk.NumericTraits[OutputPixelType].min(),
    OutputMaximum=itk.NumericTraits[OutputPixelType].max(),
)

resampler.SetDefaultPixelValue(1)
writer.SetInput(intensityRescaler.GetOutput())
writer.SetFileName(args.difference_image_after)
writer.Update()

resampler.SetTransform(identityTransform)
writer.SetFileName(args.difference_image_before)
writer.Update()