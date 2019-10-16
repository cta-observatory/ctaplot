from ctaplot.dataset import get, resources_list


def test_get_datasets():
    for file in resources_list:
        assert get(file)

