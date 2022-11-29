from omicron.models.security import Security


async def is_main_board(code):
    name = await Security.alias(code)
    if code.startswith("300") or code.startswith("688") or name.find("ST") != -1:
        return False
    return True
