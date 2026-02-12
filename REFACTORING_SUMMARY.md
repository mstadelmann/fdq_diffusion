# HDF5 Dataset Preparator Refactoring Summary

## Overview
The `hdf_preparator_v2.py` file has been completely refactored to improve code organization, readability, maintainability, and type safety.

## Key Improvements

### 1. **Code Structure & Organization**
- **Separated concerns**: Split the monolithic `H5Dataset` class into specialized classes
- **Added utility classes**: `TensorFormatter`, `HDFKeyProcessor`, `HDF5FileProcessor`, `DatasetFactory`
- **Better class hierarchy**: Each class has a single, well-defined responsibility

### 2. **Type Safety & Documentation**
- **Added type hints**: All methods now have proper type annotations
- **Improved docstrings**: Clear documentation for all classes and methods
- **Better error messages**: More descriptive error messages with context

### 3. **Configuration Management**
- **Added DatasetConfig dataclass**: Centralized configuration with default values
- **Removed magic numbers**: Configuration values are now explicit and documented
- **Better parameter handling**: Cleaner interface for dataset creation

### 4. **Code Readability**
- **Better variable names**: Replaced cryptic names like `nb_smpls_per_ds_per_grp_per_fn` with clear names
- **Shorter methods**: Broke down long methods into smaller, focused functions
- **Consistent naming**: Following Python naming conventions throughout

### 5. **Error Handling**
- **Specific exceptions**: More targeted error handling with context
- **Validation**: Better input validation with clear error messages
- **Resource management**: Proper file handling with context managers

## Class Structure

### Original Structure
```
hdf_preparator_v2.py
├── DatasetType (enum)
├── to_BCHW() function
├── to_BCDHW() function
├── _cleanup_hdf_keys() function
├── H5Dataset class (monolithic, ~500 lines)
└── create_datasets() function (monolithic, ~200 lines)
```

### Refactored Structure
```
hdf_preparator_v2.py
├── DatasetType (enum)
├── DatasetConfig (dataclass)
├── SampleInfo (dataclass)
├── TensorFormatter (utility class)
├── HDFKeyProcessor (utility class)
├── HDF5FileProcessor (processing class)
├── H5Dataset (focused dataset class)
├── DatasetFactory (creation class)
└── create_datasets() (backward compatibility function)
```

## Key Refactoring Changes

### 1. **TensorFormatter Class**
- Extracted tensor formatting logic into a dedicated class
- Improved error handling for unsupported tensor shapes
- Better documentation of tensor format assumptions

### 2. **HDFKeyProcessor Class**
- Centralized HDF key validation and processing
- Cleaner separation of key extraction and transform name handling
- Better error messages for invalid key formats

### 3. **HDF5FileProcessor Class**
- Handles all file-level processing operations
- Separate methods for different file structures (groups, datasets, pairs)
- Better progress reporting and error handling

### 4. **H5Dataset Class**
- Significantly reduced complexity (from ~500 to ~200 lines)
- Cleaner initialization process
- Better sample indexing logic
- Improved memory management

### 5. **DatasetFactory Class**
- Handles dataset creation and data loader generation
- Better configuration management
- Cleaner subsampling logic
- More comprehensive result dictionary

## Backward Compatibility

The refactored code maintains full backward compatibility:
- The `create_datasets()` function interface remains unchanged
- All existing functionality is preserved
- Configuration can still be passed through the `args` parameter

## Benefits

1. **Maintainability**: Easier to understand, modify, and extend
2. **Testability**: Each class can be tested independently
3. **Reusability**: Components can be reused in other contexts
4. **Debugging**: Clearer error messages and better logging
5. **Performance**: More efficient resource usage and better memory management
6. **Type Safety**: Fewer runtime errors due to type annotations

## Migration Guide

No changes are required for existing code that uses `create_datasets()`. The refactored implementation is a drop-in replacement that provides the same functionality with improved internal structure.

For new code, consider using the new classes directly:
```python
# New approach
config = DatasetConfig(is_3d=True, train_batch_size=64)
factory = DatasetFactory(config)
result = factory.create_datasets(experiment, args)

# Old approach (still supported)
result = create_datasets(experiment, args)
```
