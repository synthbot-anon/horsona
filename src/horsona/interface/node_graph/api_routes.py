from fastapi import APIRouter

from . import node_graph_api

api_router = APIRouter()

api_router.include_router(node_graph_api.router)
