# Cranial Suture Growth Model
This is a repository for the [Data-driven cranial suture growth model enables predicting phenotypes of craniosynostosis](https://www.nature.com/articles/s41598-023-47622-7).

The data folder contains the trained suture growth parameters (``sutureGrowthModel``) and regional weights at each anchor (``weights``) described in the manuscript. The folder also contains other necessary files to generate grwoth prediction synthesis, including the average bone segmenration maps and mask image(``averageBoneSegmentationSphericalImage`` and ``SphericalMaskImage``), and the average shape image and anatomical landmarks at birth (``InitialShapeImage`` and ``InitialLandmarks``).

This repository also provides example scripts to generate synthetic normative cranial bone surface meshes, and to simulate single suture craniosynostosis. 

## Dependencies:
- [Python](python.org)
- [NumPy](https://numpy.org/install/)
- [SimpleITK](https://simpleitk.org/)
- [VTK](https://pypi.org/project/vtk/)
- [scikit-learn](https://scikit-learn.org/stable/)

    *Once Python is installed, each of these packages can be downloaded using [Python pip](https://pip.pypa.io/en/stable/installation/)*


## Using the code

### Quick summary

**Input**: prediction_type that indicates which phenotype to simulate: takes one of the following value: 'normal', 'metopic' for metopic craniosynostosis, 'sagittal' for sagittal craniosynostosis, 'left coroanl' for left coronal craniosynostosis or 'right coronal' for right coronal craniosynostosis.

**Output**: VTK PolyData for the growth development from birth to 10 years of age, discretized in 5 days apart.


### Code example

```python
import numpy as np
import SimpleITK as sitk
import vtk
import Tools

## prediction_type takes one of the following value: 'normal', 'metopic', 'sagittal', 'left coroanl' or 'right coronal'.
prediction_type = 'normal' 

### load data
# average segmentation, intital image and mask used for transformation
anchors_per_suture = 3
base_anchor_count = 12
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

base_anchors, base_normals, base_parallel = Tools.getCranialBaseVectorsAndParallel(anchors, input_image, mask_image, base_anchor_count, anchors_per_suture)

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
```

### The workflow

- The **Tools.getAnchors** function calculates the anchors a at each suture, based on the average bone segmentation map.
- The **Tools.getIJKVectors** function calculates the suture growth vectors u_a at the sutures that is tangential to the cranial surface and perpendicular to the sutures.
- The **Tools.getCranialBaseAnchors** function calculates the anchors a at the base of the cranium.
- The **Tools.getCranialBaseVectorsAndParallel** function calculates the growth vectors u_a y_a at the cranial base.
- The **Tools.shutDownSuturalGrowth** function shut down the growth rate at specific sutures based on the phenotype to simulate.
- The **Tools.predictShapeDevelopment** function simulates the growth from birth to 10 years of age.

If you have any questions, please email Jiawei Liu at jiawei.liu@cuanschutz.edu
