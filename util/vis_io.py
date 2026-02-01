import numpy as np
import torch
import vtk
from vtkmodules.util import numpy_support
from netCDF4 import Dataset
import h5py
from pathlib import Path
from PIL import Image

def load_data(data_path, dims=None):
    raw_data_extensions = ['.raw', '.dat', '.data', '.bin']
    img_extensions = ['.png', '.jpg', '.jpeg']
    path_obj = Path(data_path)
    if path_obj.suffix in raw_data_extensions:
        data = read_raw(data_path, dims)
    elif data_path[-2:] == "nc":
        data, dims = nc_to_npy(data_path)
        data = data.squeeze()
    elif data_path[-2:] == 'h5':
        f = h5py.File(data_path)
        data = np.array(f.get('data'))
    elif path_obj.suffix in img_extensions:
        Image.MAX_IMAGE_PIXELS = None # disable DecompressionBombError
        with Image.open(data_path) as img_pil:
            data = np.array(img_pil, dtype=np.float32) / 255.0
            dims = data.shape
    # numpy array
    else:
        data = np.load(data_path)
    data = data.reshape(dims) if dims is not None else data
    return data.astype(np.float32)


def get_grid(extents, lengths):
    assert len(extents) == len(lengths)
    dim_pts = [np.linspace(ext[0], ext[1], length) for ext, length in zip(extents, lengths)]
    grid = np.meshgrid(
        *dim_pts,
        indexing="ij"
    )
    grid = np.stack(grid, -1)
    return grid


def get_grid_tensor(extent_min, extent_max, lengths, device='cpu'):
    assert len(extent_min) == len(lengths) == len(extent_max)
    dim_pts = [torch.linspace(extent_min[i], extent_max[i], steps=lengths[i], device=device) for i in range(len(lengths))]
    grid = torch.meshgrid(dim_pts, indexing="ij")
    # flip to keep spatial dimension order as X,Y,Z in the channel-first (or image DHW convention) format
    grid = torch.stack(grid[::-1], -1)
    return grid

def get_random_points(num_points, dims, device='cpu'):
    spatial_indices = [torch.randint(0, dim, (num_points,), device=device) for dim in dims.to(device)]
    spatial_indices = torch.stack(spatial_indices, dim=-1)
    return spatial_indices

def nc_to_tensor(location, opt = None, verbose=True):
    import netCDF4 as nc
    f = nc.Dataset(location)

    channels = []
    for a in f.variables:
        full_shape = f[a].shape
        d = np.array(f[a])
        channels.append(d)
    d = np.stack(channels)
    d = torch.tensor(d).unsqueeze(0)
    if verbose:
        print(f"Loaded data with shape {d.shape} (full shape: {full_shape})")
    return d, full_shape
    
def tensor_to_cdf(t: torch.Tensor, location, channel_names=None,dtype=None):
    # Assumes t is a tensor with shape (x, y, z, c)
    t = t.detach().cpu().numpy()
    import netCDF4 as nc
    d = nc.Dataset(location, 'w')

    # Setup dimensions
    d.createDimension('x')
    d.createDimension('y')
    dims = ['x', 'y']

    if(len(t.shape) == 4):
        d.createDimension('z')
        dims.append('z')

    # ['u', 'v', 'w']
    if(channel_names is None):
        ch_default = 'a'

    for i in range(t.shape[-1]):
        if(channel_names is None):
            ch = ch_default
            ch_default = chr(ord(ch)+1)
        else:
            ch = channel_names[i]
        var_dtype = np.float32 if dtype is None else dtype[i]
        d.createVariable(ch, var_dtype, dims)
        d[ch][:] = t[...,i].astype(var_dtype)
    d.close()

def nc_to_npy(location):
    f = Dataset(location)
    channels = []
    for a in f.variables:
        full_shape = f[a].shape
        d = np.array(f[a])
        channels.append(d)
    d = np.stack(channels)
    d = np.array(d)[None]
    return d, full_shape

def npy_to_nc(t, location, channel_names=None):

    print(t.shape, location)
    d = Dataset(location, 'w')
    # Setup dimensions
    d.createDimension('x')
    d.createDimension('y')
    dims = ['x', 'y']

    if(len(t.shape) == 4):
        d.createDimension('z')
        dims.append('z')

    # ['u', 'v', 'w']
    if(channel_names is None):
        ch_default = 'a'

    for i in range(t.shape[-1]):
        if(channel_names is None):
            ch = ch_default
            ch_default = chr(ord(ch)+1)
        else:
            ch = channel_names[i]
        d.createVariable(ch, np.float32, dims)
        d[ch][:] = t[...,i]
    d.close()

def read_bin(fp:str):
    '''
    read binary files with first 3 integers as dimension and floats for the rest
    '''
    c_intsize = 4
    with open(fp, 'rb') as f:
        dims = f.read(3*c_intsize)
        raw = f.read()
    dims = np.frombuffer(dims, dtype=np.int32).copy()
    return read_buffer(raw, dims)
    
def read_raw(fp:str, dims=None):
    with open(fp, 'rb') as f:
        raw = f.read()
    return read_buffer(raw, dims)

def read_buffer(buffer, dims=None):
  raw = np.frombuffer(buffer, dtype=np.float32)
  if dims is not None:
    raw = raw.reshape(dims)
  return raw.copy()

# zero-pad an axis to length
def np_zeropad(arr, length, axis):
  pad_shape = list(arr.shape)
  pad_shape[axis] = length - pad_shape[axis]
  npad = np.zeros(pad_shape)
  padded = np.concatenate((arr, npad), axis=axis)
  return padded


# create a mesh matrix of shape (D1, ..., Di, #ofDim). Di = dimension i length
def get_mesh(*dims):
  mesh_coords = []
  mesh_shape = np.array([len(dim) for dim in dims])
  for i, dim in enumerate(dims):
    # expand shape to everywhere 1 except for the dimension index
    dim_shape = np.ones(len(dims), dtype=int)
    dim_shape[i] = len(dim)
    dim_coords = dim.reshape(dim_shape)

    # repeat the length 1 dimension to match the other dimension lengths
    dim_repeats = mesh_shape.copy()
    dim_repeats[i] = 1
    dim_coords = np.tile(dim_coords, dim_repeats)
    # print(mesh_shape, hex(id(mesh_shape)), hex(id(dim_repeats)))
    mesh_coords.append(dim_coords[..., None])
  
  mesh_coords = np.concatenate(mesh_coords, axis=-1)
  print("meshe generated:", mesh_coords.shape)
  return mesh_coords


def read_raw_vti(fpath, arr_name, bbox):
  reader = vtk.vtkImageReader()
  reader.SetFileName(fpath)
  reader.SetScalarArrayName(arr_name)
  reader.SetDataByteOrderToLittleEndian()
  reader.SetDataScalarTypeToFloat()
  reader.SetDataExtent(bbox[0][0], bbox[1][0], bbox[0][1], bbox[1][1], bbox[0][2], bbox[1][2])
  reader.SetFileDimensionality(3)
  reader.SetDataOrigin(0, 0, 0)
  reader.SetDataSpacing(1, 1, 1)
  reader.Update()
  return reader.GetOutput()

def get_vti(scalar_fields={}, vector_fields={}):
    # Get the dimensions of the scalar field
    scalar_field = next(iter(scalar_fields.values()))
    dimensions = scalar_field.shape

    # Create a VTK Image Data object
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dimensions)
    
    # Set the scalar field data
    pd = image_data.GetPointData()
    for i, (k, v) in enumerate(scalar_fields.items()):
        vtk_array = numpy_support.numpy_to_vtk(v.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        vtk_array.SetName(k)
        if i == 0:
            pd.SetScalars(vtk_array)
        else:
            pd.AddArray(vtk_array)
    
    for i, (k, v) in enumerate(vector_fields.items()):
        vtk_array = numpy_support.numpy_to_vtk(v.ravel(), array_type=vtk.VTK_FLOAT)
        vtk_array.SetName(k)
        if i == 0:
            pd.SetVectors(vtk_array)
        else:
            pd.AddArray(vtk_array)
            
    return image_data

def write_vti(fpath, vti):
  writer = vtk.vtkXMLImageDataWriter()
  writer.SetFileName(fpath)
  writer.SetInputData(vti)
  writer.Write()
