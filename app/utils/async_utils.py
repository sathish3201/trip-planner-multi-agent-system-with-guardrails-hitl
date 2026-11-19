
import asyncio
import nest_asyncio

nest_asyncio.apply()


def run_async(coro):

    try:
        loop = asyncio.get_event_loop()

    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)
