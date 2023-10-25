import pytest

from e2.data_ops import DatasetIO


@pytest.mark.parametrize("path_param", [15, {}, [], set()])
def test_dataset_io_should_throw_error_if_path_not_string(path_param):
    with pytest.raises(
        TypeError, match=f"Expected data_path as str, found {type(path_param)}"
    ):
        DatasetIO(path_param)


@pytest.mark.parametrize("path_param", ["data.tsv", "data.parquet", "data.txt"])
def test_dataset_io_should_throw_error_if_ext_not_allowed(path_param):
    data_ext = path_param.split(".")[-1]

    with pytest.raises(ValueError, match=f"Expected csv file but found .{data_ext}"):
        DatasetIO(path_param)
