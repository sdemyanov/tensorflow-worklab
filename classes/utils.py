import numpy as np
import json

def dump_to_file(arrays, filename):
  arrays_for_dump = {}
  for key, array in arrays.items():
    if isinstance(array, np.ndarray):
      arrays_for_dump[key] = array.tolist()
    else:
      arrays_for_dump[key] = array
      if isinstance(array, dict):
        try:
          for k,v in array.items():
            arrays_for_dump[key][k] = v.tolist()
        except:
          pass
  with open(filename, 'w') as handle:
    json.dump(arrays_for_dump, handle, indent=2)


def load_from_file(filename):
  with open(filename, 'r') as handle:
    arrays_for_dump = json.load(handle)
  arrays = {}
  for key, array in arrays_for_dump.items():
    if isinstance(array, list):
      arrays[key] = np.asarray(array)
    elif isinstance(array, dict):
      try:
        arrays[key] = {int(k):np.asarray(v) for k,v in array.items()}
      except:
        arrays[key] = array
    else:
      arrays[key] = array
  return arrays