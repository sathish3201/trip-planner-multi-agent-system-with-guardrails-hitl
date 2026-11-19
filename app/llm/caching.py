
import litellm
from litellm.caching import Cache


def initialize_cache():

    litellm.cache = Cache(
        type="local"
    )

    return litellm.cache


cache = initialize_cache()
