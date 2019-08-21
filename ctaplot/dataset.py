import pkg_resources
import os
import sys

__all__ = ['get']


def get(resource_name):
    """ get the filename for a resource """
    try:
        resource_path = find_resource(resource_name)
    except FileNotFoundError:
        if not pkg_resources.resource_exists(__name__, resource_name):
            raise FileNotFoundError("Couldn't find resource: '{}'"
                                    .format(resource_name))
        else:
            resource_path = pkg_resources.resource_filename(__name__, resource_name)
    return resource_path


def find_resource(resource_name):
    """
    Find a resource in the share directory

    Parameters
    ----------
    resource_name: str
        name of a file to find

    Returns
    -------
    str - absolute path to the resource
    """
    # If ctaplot is installed via python setup.py develop, data files stay in share
    share_dir = os.path.join(pkg_resources.resource_filename(__name__, ''), '../share/')
    for root, dirs, files in os.walk(share_dir):
        if resource_name in files:
            return os.path.abspath(os.path.join(root, resource_name))

    # If ctaplot is installed via pip install, data files are copied in <sys.prefix>/ctaplot
    sys_dir = os.path.join(sys.prefix, 'ctaplot')
    if not os.path.exists(os.path.join(sys_dir, resource_name)):
        raise FileNotFoundError("Couldn't find resource: '{}'".format(resource_name))
    else:
        return os.path.join(sys_dir, resource_name)