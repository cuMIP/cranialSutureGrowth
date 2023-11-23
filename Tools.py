import vtk
import SimpleITK as sitk
import numpy as np
from sklearn.decomposition import PCA

def getAnchors(suture_labels, labels, anchors_per_suture=1, extremes=True):
    """returns all of the anchors for a given image based on average segmentation image

    Parameters
    ----------
    suture_labels: array_like
        shape: (6, 2)
    labels: array_like
        shape: (n, n)
    anchors_per_suture: int
        number of anchors in each suture
    extremes: bool
        whether to place the anchors at the extremes of the suture or uniformly throughout

    Returns
    -------
    anchors: array_like
        shape: (number_of_anchors, 2)
    sutures_array: array_like
        shape: (number_of_anchors, nxn, 2)
        indices of the sutures for each suture,
        if length of each suture array < n*n, then empty values are None
    """

    anchors = np.zeros((0, suture_labels.shape[1]), dtype=int)
    sutures_array = np.full((suture_labels.shape[0], labels.shape[0] * labels.shape[1], suture_labels.shape[1]), None)

    # calculating the location of the anchors
    for i in range(suture_labels.shape[0]):
        indices = np.concatenate(
            (np.argwhere(labels == suture_labels[i, 0]), np.argwhere(labels == suture_labels[i, 1]))
        )

        suture = indices[sutureLabeler(indices, labels)]
        temp = np.squeeze(np.argwhere(labels[suture[:, 0], suture[:, 1]] == suture_labels[i, 1]))
        suture = suture[temp[:]]

        # sort by y value for Metopic and Saggital sutures and squamosal?
        if i == 0 or i == 3 or i == 6 or i == 7:
            suture = suture[suture[:, 0].argsort()]

        # sort by x value for coronal and Lambdoid sutures
        else:
            suture = suture[suture[:, 1].argsort()]

        if anchors_per_suture == 1:
            temp = np.mean(suture, axis=0)
            anchors = np.append(anchors, np.expand_dims(nearestExistingIndex(temp, suture), axis=0), axis=0)
            size = suture.shape[0]
            sutures_array[i, :size] = suture[:, :]

        elif anchors_per_suture > 1 and extremes:
            # bad behavior for direction vectors at the extremums of the sutures
            suture = suture[1:-1, :]
            size = suture.shape[0]
            sutures_array[i, :size] = suture[:, :]

            anchors_one_suture = np.zeros((anchors_per_suture, suture.shape[1]))
            step = int(len(suture) / (anchors_per_suture - 1)) - 1

            for j in range(anchors_per_suture):
                anchors_one_suture[j] = suture[j * step]

            anchors = np.append(anchors, anchors_one_suture, axis=0)

        elif not extremes:
            size = suture.shape[0]
            sutures_array[i, :size] = suture[:, :]

            anchors_one_suture = np.zeros((anchors_per_suture, suture.shape[1]))
            step = int(len(suture) / (anchors_per_suture + 1))

            for j in range(anchors_per_suture):
                anchors_one_suture[j] = suture[(j + 1) * step]

            anchors = np.append(anchors, anchors_one_suture, axis=0)

    return anchors.astype(int), sutures_array

def getIJKVectors(anchors, input_image, sutures, number_of_anchors, number_of_sutures):
    """get the i, j and k direction vectors at each anchor (with relation to the suture, at the anchor)
    *** CALLS getParallelVectors() and getNormalVectors

    Parameters
    ----------
    anchors: array_like
        shape: (number of anchors, 2)
        indices of anchors in the spherical map bone image
    input_image: array_like
        shape: (n, n, 3)
        image of euclidean coordinates
    sutures: array_like
        shape: (n, m, 2)
        array of suture indices in the bone_image
    number_of_anchors: int
        number of anchors in the cranium
    number_of_sutures: int
        number of sutures in the cranium

    Returns
    -------
    (i, j, k):
        tuple containing the i, j and k vectors for all anchors
        i: array_like
            shape: (number_of_anchors, 3)
        j: array_like
            shape: (number_of_anchors, 3)
        k: array_like
            shape: (number_of_anchors, 3)
    """
    parallel_vectors = getParrallelVectors(sutures, input_image, number_of_anchors, number_of_sutures)
    normal_vectors = np.zeros(parallel_vectors.shape, dtype=np.float32)

    for n in range(number_of_anchors):
        normal_vectors[n] = getNormalVector(anchors[n], input_image)

    i = np.array(parallel_vectors)
    i = i / np.expand_dims(np.linalg.norm(i, axis=1), axis=1)
    k = normal_vectors / np.expand_dims(np.linalg.norm(normal_vectors, axis=1), axis=1)
    j = np.cross(i, k, axis=1)
    j = j / np.expand_dims(np.linalg.norm(j, axis=1), axis=1)

    return i, j, k

def getNormalVector(anchor, input_image):
    """Takes the set of pixels in a suture, and the anchor point and calculates the normal to the suture
    *** CALLED WITHIN getIJKVectors() ***

    Parameters
    ----------
    anchor: tuple
    input_image: array_like
        shape: (n, n, 3)

    Returns
    -------
    normal vector: array_like
        shape: (3,)
    """

    j, k = int(anchor[0]), int(anchor[1])

    # get two vectors, using neighbor pixels
    xpos = input_image[j + 1, k]
    xneg = input_image[j - 1, k]
    ypos = input_image[j, k + 1]
    yneg = input_image[j, k - 1]
    xvec = np.subtract(xpos, xneg)
    yvec = np.subtract(ypos, yneg)

    # calculate unit normal vector
    cross = np.cross(yvec, xvec)
    xmag = np.linalg.norm(xvec)
    ymag = np.linalg.norm(yvec)
    dot = np.dot(xvec, yvec)

    dist = 1
    while(xmag == 0):
        dist += 1
        if j + dist > input_image.shape[0] - 1:
            xpos = input_image[j + 1, k]

        else:
            xpos = input_image[j + dist, k]

        xneg = input_image[j - dist, k]
        xvec = np.subtract(xpos, xneg)
        # calculate unit normal vector
        cross = np.cross(yvec, xvec)
        xmag = np.linalg.norm(xvec)
        ymag = np.linalg.norm(yvec)
        dot = np.dot(xvec, yvec)

        dist = 1
        while(ymag == 0):
            dist += 1
            if j + dist > input_image.shape[0] - 1:
                ypos = input_image[j, k + 1]

            else:
                ypos = input_image[j, k + dist]

            yneg = input_image[j, k - dist]
            yvec = np.subtract(xpos, xneg)
            # calculate unit normal vector
            cross = np.cross(yvec, xvec)
            xmag = np.linalg.norm(xvec)
            ymag = np.linalg.norm(yvec)
            dot = np.dot(xvec, yvec)

    return np.divide(cross, np.sqrt((xmag ** 2 * ymag ** 2 - dot ** 2)))

def getParrallelVectors(sutures, input_image, number_of_anchors, number_of_sutures):
    """get the direction vectors for each anchor point in the sutures using sklearn.decomposition.PCA
    *** CALLED WITHIN getIJKVectors() ***

    Parameters
    ----------
    sutures: array_like
        shape: (n, m, 2)
        indices of the sutures in the input image
    input_image: array_like
        shape: (n, n, 3)
        spherical image with euclidean coordinates of the cranium
    number_of_anchors: int
        number of anchors in the cranium
    number_of_sutures: int
        number of sutures in the cranium

    Returns
    -------
    vectors: array_like
        shape: (number_of_anchors, 3, 3)
        the three principle directions for each anchor (i, j, k)
    """
    anchors_per_suture = int(number_of_anchors / number_of_sutures)
    vectors = np.zeros((number_of_anchors, input_image.shape[2]))

    i = 0
    for suture in sutures:
        temp_suture = suture[suture != np.array([None, None])].reshape((-1, 2))
        anchor_pixels = int(temp_suture.shape[0] / anchors_per_suture)

        for j in range(anchors_per_suture):
            pca = PCA()
            suture_segment = temp_suture[j * anchor_pixels:(j + 1) * anchor_pixels]
            suture_segment = np.array(suture_segment, dtype=int)
            suture_coords = input_image[suture_segment[:, 0], suture_segment[:, 1]]
            pca.fit(suture_coords)
            vectors[i, :] = np.array(pca.components_[0], dtype=np.float32)
            i += 1

    return vectors

def getCranialBaseAnchors(mask_image, anchor_count):
    """calculate the anchors and vectors in the cranial base

    Parameters
    ----------
    mask_image: array_like
        shape: (n, n)
        n by n binary image of the mask

    anchor_count: int
        number of anchors requested in the cranial base

    Returns
    -------
    anchors: array_like
        shape: (number_of_anchors, 2)
        matrix dimensions of the anchors in the spherical maps

    """

    indices = np.argwhere(mask_image)
    label1 = 0
    label2 = 1

    # find the neighbor pixels for each pixel
    neighbors = getNeighbors(indices)  # shape = (n, 8, 2)
    neighbor_pixels = np.zeros((neighbors.shape[0], neighbors.shape[1]))  # shape = (n, 8)
    neighbor_pixels[:, :] = mask_image[neighbors[:, :, 0], neighbors[:, :, 1]]
    pixels = mask_image[indices[:, 0], indices[:, 1]]
    pixels = np.expand_dims(pixels, axis=1)
    border = np.array([False] * len(indices))
    neighbor_pixels = np.where(neighbor_pixels[:] == -1, pixels[:], neighbor_pixels)

    # get where a pixel has a neighbor it is not equal to
    a = np.argwhere((neighbor_pixels[:] != pixels[:]))
    b = np.argwhere((neighbor_pixels[:] == label2))
    c = np.argwhere((neighbor_pixels[:] == label1))
    d = np.intersect1d(c, np.intersect1d(a, b))

    # create list of bools where borders are and index all of the indices with it
    border[d] = True
    e = np.argwhere((neighbor_pixels[:] != label1) & (neighbor_pixels[:] != label2))
    border[e] = False
    border_indices = indices[border]

    # calculate the center of the border pixels and get angles for order
    center = np.mean(border_indices, axis=0)
    angles = np.arctan2(border_indices[:, 0] - center[0], border_indices[:, 1] - center[1])
    # sorted = np.array(angles)
    # sorted = (sorted * 180) / np.pi
    angles = (angles * 180) / np.pi
    # sorted[::1].sort()

    center = np.array([33,41])
    angles = np.arctan2(border_indices[:, 0] - center[0], border_indices[:, 1] - center[1])
    sorted = np.array(angles)
    sorted = (sorted * 180) / np.pi
    angles = (angles * 180) / np.pi
    sorted[::1].sort()

    (sorted<0).sum()

    angles[angles<0] = 180- angles[angles<0]
    anchors_test = np.zeros((anchor_count, 2))
    # anchors_test[0] = [border_indices[np.argmin(np.abs(angles)),0], border_indices[np.argmin(np.abs(angles)),1]]
    # anchors_test[1] = [border_indices[np.argmax(np.abs(angles)),0], border_indices[np.argmax(np.abs(angles)),1]]

    for i in range(12):
        anchors_test[i] = [border_indices[np.where(np.abs(angles - i * 360/12)==np.abs(angles - i * 360/12).min())[0][0],0], border_indices[np.where(np.abs(angles - i * 360/12)==np.abs(angles - i * 360/12).min())[0][0],1]]

    anchors_test = anchors_test.astype(np.int8)

    return anchors_test

def getNeighbors(indices):
    """identifies the 8 neighbor pixels for a given pixel

    Parameters
    ----------
    indices: array_like
        shape: (n, 2)
        storing indicies of pixels that need neighbors returned

    Returns
    -------
    neighbors: array_like
        shape: (n * 8, 2)
        array of indices of the neighboring pixels to pixel
    """

    a = indices[:, 0]
    b = indices[:, 1]
    neighbors = np.array(
        [
            (a - 1, b - 1), (a - 1, b), (a - 1, b + 1), (a, b - 1),
            (a, b + 1), (a + 1, b - 1), (a + 1, b), (a + 1, b + 1)
        ], dtype=np.int
    )

    neighbors = np.transpose(neighbors, (2, 0, 1))
    return neighbors

def getCranialBaseVectors(landmarks, anchors, input_image):
    """calculate the vectors normal to the cranial surface at the cranial base

    Parameters
    ----------
    landmarks: array_like
        shape: (4, 2)
        the four landmarks in the base of the cranium
    anchors: array_like
        shape: (m, 2)
        the cranial base anchors in the spherical maps
    input_image: array_like
        shape: (n, n, 3)
        n x n euclidean coordinates image

    Return
    ------
    array_like, shape: (anchors.shape[0], 3)
    """
    centers = input_image[anchors[:, 0], anchors[:, 1]]
    vectors = centers - np.mean(landmarks[1:3])
    vectors /= np.expand_dims(np.linalg.norm(vectors, axis=1), axis=1)

    return vectors

def getCranialBaseVectorsTest(anchors, input_image, mask_image, base_anchor_count, anchors_per_suture):
    """calculate the vectors normal to the cranial surface at the cranial base

    Parameters
    ----------
    landmarks: array_like
        shape: (4, 2)
        the four landmarks in the base of the cranium
    anchors: array_like
        shape: (m, 2)
        the cranial base anchors in the spherical maps
    input_image: array_like
        shape: (n, n, 3)
        n x n euclidean coordinates image

    Return
    ------
    array_like, shape: (anchors.shape[0], 3)
    """
    base_anchors = getCranialBaseAnchors(mask_image, base_anchor_count)
    # get landmarks to correct cranial base vector directions
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName('InitialLandmarks.vtp')
    reader.Update()
    landmarkPolyData = reader.GetOutput()
    landmarks = np.array(landmarkPolyData.GetPoints().GetData())
    
    centers = input_image[base_anchors[:, 0], base_anchors[:, 1]]
    base_normals = centers - np.mean(landmarks[1:3])
    base_normals /= np.expand_dims(np.linalg.norm(base_normals, axis=1), axis=1)

    top_base_anchor = np.squeeze(np.argwhere(np.amax(base_anchors[:, 0]) == base_anchors[:, 0]))
    bottom_base_anchor = np.squeeze(np.argwhere(np.amin(base_anchors[:, 0]) == base_anchors[:, 0]))

    base_normals[top_base_anchor] = (landmarks[0] - np.mean(landmarks[1:3], axis=0))
    base_normals[top_base_anchor] /= np.linalg.norm(base_normals[top_base_anchor])
    base_normals[bottom_base_anchor] = (landmarks[3] - np.mean(landmarks[1:3], axis=0))
    base_normals[bottom_base_anchor] /= np.linalg.norm(base_normals[bottom_base_anchor])
    base_normals[:-4] = base_normals[bottom_base_anchor]
    base_normals = base_normals[base_anchors[:, 0].argsort()]

    base_anchors = base_anchors[base_anchors[:, 0].argsort()]

    for i in range(base_anchors[:int(base_anchor_count * 0.75)].shape[0]):
        base_normals[i] = input_image[base_anchors[i, 0], base_anchors[i, 1]] - np.mean(landmarks[1:3], axis=0)
        base_normals[i] /= np.linalg.norm(base_normals[i])

    ## normal to the cutting planes
    center = np.mean(input_image[mask_image==1], axis = 0)
    landmarkCenter = np.mean(landmarks, axis = 0)
    dorsumSella = landmarks[2]- landmarks[1]
    v1 = center - landmarkCenter
    base_parallel = np.zeros(base_normals.shape, dtype = base_normals.dtype)
    for pt in range(base_normals.shape[0]):
        if pt != (base_normals.shape[0]-1):
            base_normals[pt,: ] = input_image[base_anchors[pt,0], base_anchors[pt,1], :] - center
            # vector_parallel = input_image[base_anchors[pt, 0], base_anchors[pt, 1], :] - input_image[base_anchors[pt+1, 0], base_anchors[pt+1, 1], :]
            # base_parallel[pt,:] = np.cross(base_normals[pt], vector_parallel)
            v2 = np.cross(v1, base_normals[pt,:])
            base_parallel[pt,:] = np.cross(v2, base_normals[pt,:])
            base_normals[pt,: ] = base_normals[pt,:] / np.linalg.norm(base_normals[pt,:])
            base_parallel[pt,:] = base_parallel[pt,:] / np.linalg.norm(base_parallel[pt,:])
            if np.dot(input_image[base_anchors[pt, 0], base_anchors[pt, 1], :] - input_image[anchors[3 * anchors_per_suture, 0], anchors[3 * anchors_per_suture, 1], :], base_parallel[pt,:]) > 0:
                base_parallel[pt,:] = - base_parallel[pt,:]
        else:
            base_normals[pt,: ] = input_image[base_anchors[pt,0], base_anchors[pt,1], :] - center
            # vector_parallel = input_image[base_anchors[pt, 0], base_anchors[pt, 1], :] - input_image[base_anchors[pt+1, 0], base_anchors[pt+1, 1], :]
            # base_parallel[pt,:] = np.cross(base_normals[pt], vector_parallel)
            v2 = np.cross(v1, base_normals[pt,:])
            base_parallel[pt,:] = np.cross(v2, base_normals[pt,:])
            base_normals[pt,: ] = base_normals[pt,:] / np.linalg.norm(base_normals[pt,:])
            base_parallel[pt,:] = base_parallel[pt,:] / np.linalg.norm(base_parallel[pt,:])
            if np.dot(input_image[base_anchors[pt, 0], base_anchors[pt, 1], :] - input_image[anchors[3 * anchors_per_suture, 0], anchors[3 * anchors_per_suture, 1], :], base_parallel[pt,:]) > 0:
                base_parallel[pt,:] = - base_parallel[pt,:]

    return base_anchors, base_normals, base_parallel

def sutureLabeler(indices, bone_image):
    """labels the sutures of a segmentation image

    Parameters
    ----------
    indices: array_like
        shape: (n, 2)
        indices of pixels of two bones
    bone_image: array_like
        shape: (n, n)
        bone labels

    Returns
    -------
    sutures: array_like
        shape: (n, 2)
        array of booleans for pixels in the sutures, true if inside, false if outside
    """

    # two bone labels for a given suture
    label1 = bone_image[indices[0, 0], indices[0, 1]]
    label2 = bone_image[indices[indices.shape[0] - 1, 0], indices[indices.shape[0] - 1, 1]]

    # find the neighbor pixels for each pixel
    neighbors = getNeighbors(indices)  # shape = (n, 8, 2)
    neighbor_pixels = np.zeros((neighbors.shape[0], neighbors.shape[1]))  # shape = (n, 8)
    neighbor_pixels[:, :] = bone_image[neighbors[:, :, 0], neighbors[:, :, 1]]
    pixels = bone_image[indices[:, 0], indices[:, 1]]
    pixels = np.expand_dims(pixels, axis=1)
    sutures = np.array([False] * len(indices))
    neighbor_pixels = np.where(neighbor_pixels[:] == -1, pixels[:], neighbor_pixels)

    a = np.argwhere((neighbor_pixels[:] != pixels[:]))
    b = np.argwhere((neighbor_pixels[:] == label2))
    c = np.argwhere((neighbor_pixels[:] == label1))
    d = np.intersect1d(c, np.intersect1d(a, b))

    sutures[d] = True

    e = np.argwhere((neighbor_pixels[:] != label1) & (neighbor_pixels[:] != label2))

    sutures[e] = False

    return sutures

def ConstructCranialSurfaceMeshFromSphericalMaps(euclideanCoordinateSphericalMapImage, referenceImage, intensityImageDict=None, subsamplingFactor=5, verbose=False):
    """
    Recsontructs a surface model using the Euclidean coordinates represented as a spherical map image 

    Parameters
    ----------
    euclideanCoordinateSphericalMapImage: sitkImage
        Spherical map image with the Euclidean coordinates of the surface model
    referenceImage: sitkImage
        A reference image with with pixels set to 0 in the background
    intensityImageDict: dictionary {arrayName: image}
        A dictionary

    Returns
    -------
    vtk.vtkPolyData:
        The reconstructed mesh    
    """

    bullsEyeImageArray = sitk.GetArrayViewFromImage(euclideanCoordinateSphericalMapImage)
    referenceImageArray = sitk.GetArrayViewFromImage(referenceImage)

    filter = vtk.vtkPlaneSource()
    filter.SetOrigin((-1, -1, 0))
    filter.SetPoint1((1, -1, 0))
    filter.SetPoint2((-1, 1, 0))
    filter.SetXResolution(int(euclideanCoordinateSphericalMapImage.GetSize()[0] / subsamplingFactor))
    filter.SetYResolution(int(euclideanCoordinateSphericalMapImage.GetSize()[1] / subsamplingFactor))
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkTriangleFilter()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkCleanPolyData()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()

    insideArray = vtk.vtkIntArray()
    insideArray.SetName('Inside')
    insideArray.SetNumberOfComponents(1)
    insideArray.SetNumberOfTuples(mesh.GetNumberOfPoints())
    mesh.GetPointData().AddArray(insideArray)


    intensityImageArrays = []
    intensityArrays = []
    if intensityImageDict is not None:

        for key, val in intensityImageDict.items():

            intensityImageArrays += [sitk.GetArrayViewFromImage(val)]
            intensityArrays += [vtk.vtkFloatArray()]
            intensityArrays[-1].SetName(key)
            intensityArrays[-1].SetNumberOfComponents(1)
            intensityArrays[-1].SetNumberOfTuples(mesh.GetNumberOfPoints())
            mesh.GetPointData().AddArray(intensityArrays[-1])

    # Figuring out what is inside or outside
    for p in range(mesh.GetNumberOfPoints()):
            
        if verbose:
            print('{} / {}.'.format(p, mesh.GetNumberOfPoints()), end='\r')

        coords = mesh.GetPoint(p)

        try:
            imageIndex = referenceImage.TransformPhysicalPointToIndex((coords[0], coords[1]))

            mesh.GetPoints().SetPoint(p, euclideanCoordinateSphericalMapImage.GetPixel(imageIndex))

            if referenceImageArray[imageIndex[1], imageIndex[0]] > 0:
                insideArray.SetTuple1(p, 1)
                
                if intensityImageDict is not None:

                    for arrayId in range(len(intensityArrays)):
                        intensityArrays[arrayId].SetTuple1(p, intensityImageArrays[arrayId][imageIndex[1], imageIndex[0]])
            else:
                insideArray.SetTuple1(p, 0)
        except:
            insideArray.SetTuple1(p, 0)

    filter = vtk.vtkThreshold()
    filter.SetInputData(mesh)
    filter.ThresholdByUpper(0.9)
    filter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 'Inside')
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkGeometryFilter()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(mesh)
    filter.ComputeCellNormalsOn()
    filter.ComputePointNormalsOn()
    filter.NonManifoldTraversalOff()
    filter.AutoOrientNormalsOn()
    filter.ConsistencyOn()
    filter.SplittingOff()
    filter.Update()
    mesh = filter.GetOutput()

    mesh.GetPointData().RemoveArray("Inside")

    return mesh

def transformShape(
    transformed_points, timepoint, delta, scale_parameters, anchors, centers, weights, vectors, base_anchor_count
):
    """recursive definition of transformation

    Parameters
    ----------
    transformed_points: array_like
        shape: (n, n, 3)
        transformed points at time t-1 (original points when t-1 = 0)

    timepoint: float
        t / (N-1)

    delta: float
        1 / (N-1)

    scale_parameters: array_like
        shape: (anchors.shape[0] * P.shape[0],)
        parameters used to calculate velocities

    anchors: array_like
        shape: (number of anchors, 2)
        indices of the anchors within _points

    centers: array_like
        shape: (number of anchors, 3)
        euclidean coordinates of centers of transformation

    derivative_centers: array_like
        shape: (number of parameters, 3)
        euclidean coordinates of centers of transformation

    weights: array_like
        shape: (anchors.shape[0], n, n, 3)
        weights based on gaussian kernel used to weight the transformation at each anchor

    vectors: array_like
        shape: (anchors.shape[0], 3)
        vectors with the growth direction

    base_anchor_count: int
        number of translation parameters in the cranial base

    Returns
    -------
    tuple: transformed points, derivatives of the transformation, centers, and derivative centers
    """
    def velocity(a, b, c, t):
        return(np.exp(a) / (np.exp(b)*t + np.exp(c)*t**2 + 1))

    previous_transformed_points = transformed_points

    ### transformed points ###
    moving_points_minus_centers = np.zeros(
        (anchors.shape[0], transformed_points.shape[0], transformed_points.shape[1], transformed_points.shape[2]),
        dtype=np.float32
    )

    # structure to hold (1 + v(t-1) * delta_t)
    scale = np.zeros((anchors.shape[0], vectors.shape[1]), dtype=np.float32)

    # calculate velocity at each anchor
    scaled_points_minus_centers = np.zeros(moving_points_minus_centers.shape, dtype=np.float32)

    for a in range(anchors.shape[0]):
        moving_points_minus_centers[a, :, :, :] = previous_transformed_points - centers[a:a + 1, :] # T(x, t-1, p) - T(c_a, t-1, p)
        vel = velocity(
            scale_parameters[int(scale_parameters.shape[0] / anchors.shape[0]) * a],
            scale_parameters[int(scale_parameters.shape[0] / anchors.shape[0]) * a + 1],
            scale_parameters[int(scale_parameters.shape[0] / anchors.shape[0]) * a + 2],
            timepoint
        ) # v(t-1)

        if a < (anchors.shape[0] - base_anchor_count):
  
            scale[a] = vel * delta * vectors[a]

            scaling  = np.zeros((transformed_points.shape[0], transformed_points.shape[1], transformed_points.shape[2]), dtype=np.float64)
            for axis in range(3):
                mag = np.dot(moving_points_minus_centers[a, :, :, :], vectors[a])
                mag = np.array(mag, dtype = np.float32)
                scaling[:,:,axis] = scale[a, axis] * mag

            scaled_points_minus_centers[a] = scaling + previous_transformed_points

        else:
            
            scale[a] = vel * delta * vectors[a]
            scaled_points_minus_centers[a, :, :, :] = previous_transformed_points + scale[a]

    scaled_points_minus_centers = scaled_points_minus_centers * weights

    # sum points along anchors axis
    transformed_points = np.sum(scaled_points_minus_centers, axis=0)

    # get new centers and derivative centers for next transformation
    centers = transformed_points[anchors[:, 0], anchors[:, 1], :]

    return transformed_points, centers

def predictShapeDevelopment(X, increments, anchors, centers, weights, scale_parameters, vectors, base_anchor_count):
    transformed_points = X

    delta = 1.0 / (increments - 1)
    # number_of_suture_anchors = anchors.shape[0] - base_anchor_count

    transformed_points_structure = np.empty((increments, X.shape[0], X.shape[1], X.shape[2]))
    for i in range(increments):
        timepoint = i/increments

        transformed_points, centers = transformShape(
            transformed_points, timepoint, delta,
            scale_parameters, anchors, centers, weights, vectors, base_anchor_count
        )
        transformed_points_structure[i] = transformed_points

    return transformed_points_structure
