from fastapi import APIRouter

from . import oai_api

api_router = APIRouter()

api_router.include_router(oai_api.router)
