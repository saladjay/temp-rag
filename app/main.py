"""FastAPI 应用工厂。"""
from fastapi import FastAPI
from app.api.routes import router, health_router


def create_app(testing: bool = False) -> FastAPI:
    """创建并配置 FastAPI 应用实例。

    :param testing: 测试标记（v1 预留，当前不影响行为）。
    :return: FastAPI 应用。
    """
    app = FastAPI(title="多轮对话智能客服")
    app.include_router(router)
    app.include_router(health_router)
    return app


# 模块级实例：供 uvicorn app.main:app 启动
app = create_app()
