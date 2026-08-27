from .auth import router as auth_router
from .users import router as users_router
from .movies import router as movies_router
from .recommendations import router as recommendations_router
from .rooms import router as rooms_router
from .runs import router as runs_router

__all__ = [
    "auth_router",
    "users_router",
    "movies_router",
    "recommendations_router",
    "rooms_router",
    "runs_router",
]
