from ctaplot.io.dataset import get, resources_list, load_any_resource


def test_get_datasets():
    for file in resources_list:
        assert get(file)


def test_resources_list():
    for filename in resources_list:
        get(filename)


def test_load_any_resource():
    for filename in resources_list:
        data = load_any_resource(filename)
        assert len(data) > 0