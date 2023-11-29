import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
import vtk
import matplotlib.pyplot as plt
import Tools

anchors_per_suture = 3
base_anchor_count = 12
## prediction_type takes one of the following value: 'normal', 'metopic', 'sagittal', 'left coroanl' or 'right coronal'.
prediction_type = 'normal' 

### load data
# average segmentation, intital image and mask used for transformation
bones = sitk.ReadImage('data/averageBoneSegmentationSphericalImage.mha')
age0 = sitk.ReadImage('data/InitialShapeImage.mha')
mask = sitk.ReadImage('data/SphericalMaskImage.mha')

# put images in numpy data structure and sample pizels
sample = 6 # subsampling factor
mask_image = sitk.GetArrayFromImage(mask)[::6, ::6]
input_image = sitk.GetArrayFromImage(age0)[::6, ::6]
bone_image = sitk.GetArrayFromImage(bones)[::6, ::6]

# get indices of mask to limit aligned image
mask_indices = np.argwhere(mask_image == 0)
indices = np.argwhere(mask_image == 1)

# mask out background
input_image[mask_indices[:, 0], mask_indices[:, 1]] = 0
bone_image[mask_indices[:, 0], mask_indices[:, 1]] = 0

# labels of bone regions in the segmentation image that contain sutures in the calvaria
suture_labels = np.array([(1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5), (3, 6), (4, 7)])

# get anchors vectors and weights
anchors, sutures  = Tools.getAnchors(suture_labels, bone_image, anchors_per_suture, extremes=True)
_, vectors, normals = Tools.getIJKVectors(anchors, input_image, sutures, anchors.shape[0], suture_labels.shape[0])

base_anchors = Tools.getCranialBaseAnchors(mask_image, base_anchor_count)

base_anchors, base_normals, base_parallel = Tools.getCranialBaseVectorsTest(anchors, input_image, mask_image, base_anchor_count, anchors_per_suture)

anchors = np.concatenate((np.concatenate((anchors, base_anchors), axis=0), base_anchors), axis = 0)
vectors = np.concatenate((np.concatenate((vectors, base_parallel), axis=0), base_normals), axis = 0)
centers = input_image[anchors[:, 0], anchors[:, 1], :]

weights = np.load('data/weights.npy')

#### predict growth
scale_parameters = np.load('data/sutureGrowthModelExponentiated.npy')
scale_parameters = Tools.shutDownSuturalGrowth(scale_parameters, prediction_type)
increments = int(3625.25/5)
transformed_points_structure = Tools.predictShapeDevelopment(
    input_image, increments, anchors, centers, weights, scale_parameters, vectors, base_anchor_count
)

## visualize weights

sampled_mask = sitk.GetImageFromArray(mask_image, isVector=True)
sampled_mask.SetOrigin((-1., -1.))
sampled_mask.SetSpacing((sample * age0.GetSpacing()[0], sample * age0.GetSpacing()[0]))

for i in range(transformed_points_structure.shape[0]):
    print(i)
    image = sitk.GetImageFromArray(transformed_points_structure[i], isVector=True)
    image.SetOrigin((-1., -1.))
    image.SetSpacing((sample * age0.GetSpacing()[0], sample * age0.GetSpacing()[0]))

    original_mesh = Tools.ConstructCranialSurfaceMeshFromSphericalMaps(
        image, sampled_mask, intensityImageDict={}, subsamplingFactor=2 / sample, verbose=False
    )
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("shapeAtAge{:.2f}Years.vtp".format(i/increments * 10))
    writer.SetInputData(original_mesh)
    writer.Update()
