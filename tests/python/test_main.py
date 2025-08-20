import pytest

def test_import_module():
    import gsm_data_generator
    assert gsm_data_generator is not None
    
    # Ensure version exists and is a string
    assert hasattr(gsm_data_generator, "__version__")
    assert isinstance(gsm_data_generator.__version__, str)
    print("gsm_data_generator version:", gsm_data_generator.__version__)
