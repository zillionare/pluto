from omicron.models.security import Security


async def is_main_board(code):
    name = await Security.alias(code)
    if code[:3] in ("300", "301", "688", "689") or name.find("ST") != -1:
        return False
    return True
