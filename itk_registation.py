import argparse
import itk
import cv2
import os
from config import output_folder
import SimpleITK as sitk
from config import output_folder
import numpy as np
from utils import whitewashing, otsu

# if VS(itk.Version.GetITKVersion()) < VS("4.9.0"):
#     print("ITK 4.9.0 is required.")
#     sys.exit(1)


def itk_registration(fixed_input_image,
                     moving_input_image,
                     output_name="output"):
    PixelType = itk.ctype("float")

    # Convert images so that there is no transparent background + good itk compatibility
    fixedImage = cv2.imread(fixed_input_image.as_posix())
    movingImage = cv2.imread(moving_input_image.as_posix())
    fixedImage, movingImage = whitewashing(fixedImage), whitewashing(movingImage)
    fixedImage, movingImage = otsu(fixedImage), otsu(movingImage)
    cv2.imwrite('tmp1.png', fixedImage)
    cv2.imwrite('tmp2.png', movingImage)
    fixedImage = itk.imread('tmp1.png', PixelType)
    movingImage = itk.imread('tmp2.png', PixelType)
    os.remove("tmp1.png")
    os.remove("tmp2.png")

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

    writer = itk.ImageFileWriter.New(Input=caster, FileName=(output_folder / f"{output_name}_final.png").as_posix())
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
    writer.SetFileName((output_folder / f"{output_name}_after.png").as_posix())
    writer.Update()

    resampler.SetTransform(identityTransform)
    writer.SetFileName((output_folder / f"{output_name}_before.png").as_posix())
    writer.Update()

    return translationAlongX / movingImage.shape[1], translationAlongY / movingImage.shape[0]