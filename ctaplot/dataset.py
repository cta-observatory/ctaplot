import pkg_resources

__all__ = ['get']


def get(resource_name):
    """ get the filename for a resource """
    if not pkg_resources.resource_exists(__name__, resource_name):
        raise FileNotFoundError("Couldn't find resource: '{}'"
                                .format(resource_name))
    return pkg_resources.resource_filename(__name__, resource_name)
